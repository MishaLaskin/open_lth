# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
import copy 
import torch

class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    class AuxBlock(nn.Module):
        def __init__(self,pool_dim,lin_dim,num_classes=10):
            super(Model.AuxBlock, self).__init__()
            self.pool = nn.AvgPool2d(pool_dim)
            self.linear = nn.Linear(lin_dim,num_classes)

        def forward(self, x):
            out = self.pool(x) 
            B, C, H, W = out.shape 
            out = out.view(B,-1)
            out = self.linear(out)
            return out

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        self.first_block = nn.Sequential(self.conv,self.bn)

        # The subsequent blocks of the ResNet.
        blocks = [self.first_block]
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        

        self.blocks = nn.ModuleList(blocks)

        self.ema = True

        self.ema_blocks = copy.deepcopy(self.blocks)
        # set ema blocks to have same params as blocks
        self.ema_blocks.load_state_dict(self.blocks.state_dict())

        self.aux_blocks = self._make_auxs()

        self.cache = [None] * len(self.blocks)
        self.label_cache = [None] * len(self.blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    

    def _make_auxs(self):

        aux_blocks = nn.ModuleList([])
        x = torch.randn(1, 3, 32, 32)
    
        for i, b in enumerate(self.blocks):
            x = b(x)
            B, C, H, W = x.shape
            flat_dim = C*H*W

            if flat_dim > 30000 or i == len(self.blocks) - 1:
                x_aux = F.avg_pool2d(x, 4)
                lin_dim = x_aux.view(B,-1).shape[-1]
                aux_block = Model.AuxBlock(4,lin_dim)
            elif flat_dim > 15000:
                x_aux = F.avg_pool2d(x, 2)
                lin_dim = x_aux.view(B,-1).shape[-1]
                aux_block = Model.AuxBlock(2,lin_dim)
            else:
                x_aux = x 
                lin_dim = x_aux.view(B,-1).shape[-1]
                aux_block = Model.AuxBlock(1,lin_dim)
                
            aux_blocks.append(aux_block)
        return aux_blocks

    def single_forward(self,x,layer):
        m = layer
        
        if m > 0:
            # takes last cached value and detached gradient
            # so that only m-th layer is updated
            x = self.cache[m-1].clone().detach() # takes input from ema cache
        # get output to cache later
        x_cache = self.ema_blocks[m](x) if self.ema else self.blocks[m](x)
        self.cache[m] = x_cache.clone().detach()
        
        x = self.blocks[m](x) 
        B, C, G, W = x.shape
        y = self.aux_blocks[m](x)
            
        return y

    def forward_train_overlap(self,x,layer):
        m = layer
        not_last = True if m != len(self.blocks) - 1 else False
        if m > 0:
            # takes last cached value and detached gradient
            # so that only m-th layer is updated
            x = self.cache[m-1].clone().detach() # takes input from ema cache
        # get output to cache later
        x_cache =  self.ema_blocks[m](x) if self.ema else self.blocks[m](x)
        self.cache[m] = x_cache.clone().detach()
        
        x = self.blocks[m](x) # x_{t-1} 
        if not_last:
            x = self.blocks[m+1](x) # x_t 
            y = self.aux_blocks[m+1](x)
        else:
            y = self.aux_blocks[m](x)

        return y

    def forward_train_greedy(self,x,layer):
        m = layer
        not_last = True if m != len(self.blocks) - 1 else False
        if m > 0:
            # takes last cached value and detached gradient
            # so that only m-th layer is updated
            x = self.cache[m-1].clone().detach() # takes input from ema cache
        # get output to cache later
        x_cache =  self.ema_blocks[m](x) if self.ema else self.blocks[m](x)
        self.cache[m] = x_cache.clone().detach()
        
        x = self.blocks[m](x) # x_{t-1} 
        y = self.aux_blocks[m](x)

        return y

    def forward(self,x):
        for j in range(len(self.blocks)):
        
            outputs = self.single_forward(x,j)
        return outputs 

    def compute_train_loss(self, x, targets, greedy=True):
    
        fwd_pass = self.forward_train_greedy if greedy else self.forward_train_overlap
        #train_losses = [0 for _ in range(len(net.blocks))]
        #corrects = [0 for _ in range(len(net.blocks))]
        loss = None
        for j in range(len(self.blocks)):
            outputs = fwd_pass(x,layer=j)
            local_loss = self.criterion(outputs, targets) 
        
            if loss == None:
                loss = local_loss
            else:
                loss += local_loss
            #train_losses[j] += local_loss.item()
            #_, predicted = outputs.max(1)
            #corrects[j] += predicted.eq(targets).sum().item()

        return loss



    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        
        return (model_name.startswith('cifar_localresnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """The naming scheme for a ResNet is 'cifar_localresnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_localresnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_local_resnet_20' or 'cifar_local_resnet_20_16'.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_localresnet_20',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)

from functools import partial
import os
import sys
import logging
import functools
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("lib/models")
from mdeq_core import MDEQNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ContrastiveLinearMDEQ(MDEQNet):
    """ implements contrastive loss https://arxiv.org/abs/2002.05709 """

    def __init__(self, cfg, **kwargs):
        """ init additional BN used after head """
        super(ContrastiveLinearMDEQ, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.emb_size = cfg['CONTRASTIVE']['EMB_SIZE']
        self.head_layers = cfg['CONTRASTIVE']['HEAD']['LAYERS']
        self.head_size = cfg['CONTRASTIVE']['HEAD']['SIZE']
        self.add_bn = cfg['CONTRASTIVE']['HEAD']['ADD_BN']
        self.heads = [self.build_head(in_size) for in_size in cfg.CONTRASTIVE.REPR_SIZE]

    def build_head(self, in_size):
        """ creates projection head g() from config """
        x = []
        for _ in range(self.head_layers - 1):
            x.append(nn.Linear(in_size, self.head_size))
            if self.add_bn:
                x.append(nn.BatchNorm1d(self.head_size))
            x.append(nn.ReLU())
            in_size = self.head_size
        x.append(nn.Linear(in_size, self.emb_size))
        if self.add_bn:
            x.append(nn.BatchNorm1d(self.emb_size))
        return nn.Sequential(*x)

    def forward(self, x, train_step=0, project=False, **kwargs):
        out = self._forward(x, train_step, **kwargs)

        # only take the lowest resolution representation, we can fix this later to take all of them!
        for i in range(len(out)):
            out[i] = out[i].flatten(start_dim=1)

        # project by default (throw away later)
        if project:
            for i in range(len(self.heads)):
                out[i] = self.heads[i](out[i])

        return out

    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class ContrastiveDownsampleMDEQ(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQClsNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']
        self.frozen = cfg.MODEL.FROZEN

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.projection = nn.Linear(self.final_chansize, self.emb_size)

    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels

        # Increasing the number of channels on each resolution when doing classification.
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
                                              self.final_chansize,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, train_step=0, project=False, **kwargs):

        if self.frozen:
            with torch.no_grad():
                y_list = self._forward(x, train_step, **kwargs)
        else:
            y_list = self._forward(x, train_step, **kwargs)

        if project:
            # Classification Head
            y = self.incre_modules[0](y_list[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
            y = self.final_layer(y)

            # Pool to a 1x1 vector (if needed)
            if torch._C._get_tracing_state():
                y = y.flatten(start_dim=2).mean(dim=2)
            else:
                y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
            y = self.projection(y)

        else:
            y = [y_val.flatten(start_dim=1) for y_val in y_list]

        return y

    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_contrastive_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = ContrastiveLinearMDEQ(config, **kwargs)
    model.init_weights()
    return model

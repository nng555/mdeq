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

class ContrastiveMDEQ(MDEQNet):
    """ implements contrastive loss https://arxiv.org/abs/2002.05709 """

    def __init__(self, cfg, **kwargs):
        """ init additional BN used after head """
        super(ContrastiveMDEQ, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.emb_size = cfg['CONTRASTIVE']['EMB_SIZE']
        self.out_size = cfg['CONTRASTIVE']['REPR_SIZE']
        self.head_layers = cfg['CONTRASTIVE']['HEAD']['LAYERS']
        self.head_size = cfg['CONTRASTIVE']['HEAD']['SIZE']
        self.add_bn = cfg['CONTRASTIVE']['HEAD']['ADD_BN']
        self.head = self.build_head(cfg)

    def build_head(self, cfg):
        """ creates projection head g() from config """
        x = []
        in_size = self.out_size
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

    def forward(self, x, train_step=0, project=True, **kwargs):
        out = self._forward(x, train_step, **kwargs)

        # only take the lowest resolution representation, we can fix this later to take all of them!
        out = out[-1]
        out = out.flatten(start_dim=1)

        # project by default (throw away later)
        if project:
            out = self.head(out)
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

def get_contrastive_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = ContrastiveMDEQ(config, **kwargs)
    model.init_weights()
    return model

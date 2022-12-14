# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch import nn
from torch.nn import functional
import torch
# from dense_deep_event_stereo import continuous_fully_connected
from model_LTC import network_blocks

def convolution_3x1x1_with_relu(number_of_input_features,
                                number_of_output_features):
    return nn.Sequential(
        nn.Conv3d(number_of_input_features,
                  number_of_output_features,
                  kernel_size=(3, 1, 1),
                  padding=(1, 0, 0)),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class ConvLTC(nn.Module):
    '''no intra-layer recurrent connection'''
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
        in_channels, num_features, tau_input, taum_ini, usetaum, stream_opt, self.burn_in_time = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum'], hparams['stream_opt'], hparams['burn_in_time']
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv = self._make_layer(in_channels, num_features, kernel_size, padding, stride)
        self.usetaum = usetaum    
        self.stream_opt = stream_opt
        self.cm = nn.Parameter((0.4-0.6)*torch.rand(num_features,1,1)+0.6)
        self.vleak = nn.Parameter((-0.2-0.2)*torch.rand(num_features,1,1)+0.2)
        if self.usetaum:
            self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])

        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     
        
        self._epsilon = 1e-8
        
        self.sigmoid = nn.Sigmoid()
        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None

        nn.init.xavier_normal_(self.conv[0].weight.data)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    #def init_v_pre(self, B, num_features, H, W):
    #    self.v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self.cm.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
        else:
            self.gleak.data.clamp_(0,1000)
    
    def forward(self, inputs, is_train):
        '''
        :param inputs: (B, C_in, S, H, W)
        :param hidden_state: (hx: (B, C, S, H, W), cx: (B, C, S, H, W))
        :return: (B, C_out, H, W)
        '''
        B, C, S, H, W = inputs.size()
        outputs = []
        cm_t = self.cm 
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):     

            wih = self.conv(inputs[:, t]) # wi*sig(x)+wh*sig(vpre)

            if self.tau_input:
                if self.usetaum:
                    numerator = (
                        cm_t * v_pre
                        + (self.cm / self.tau_m) * self.vleak
                        + wih*self.E_revin
                    )
                    denominator = cm_t + (self.cm/self.tau_m) + wih
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih*self.E_revin
                    )
                    denominator = cm_t + self.gleak + wih

            else:
                if self.usetaum:

                    numerator = (
                        cm_t * v_pre
                        + (self.cm / self.tau_m) * self.vleak
                        + wih# *self.E_revin
                    )
                    denominator = cm_t + self.cm / self.tau_m 
    
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih
                    )
                    denominator = cm_t + self.gleak


            v_pre = numerator / (denominator + self._epsilon)

            v_pre = self.sigmoid(v_pre)

            outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:   
            return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        else:
            return outputs[-1]

class ConvLTC_v1(nn.Module):
    '''more general discrete form of LTC'''  
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
        in_channels, num_features, tau_input, taum_ini, usetaum, stream_opt, self.burn_in_time = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum'], hparams['stream_opt'], hparams['burn_in_time']
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv = self._make_layer(in_channels, num_features, kernel_size, padding, stride)
        self.usetaum = usetaum    
        self.stream_opt = stream_opt
        self.cm = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        self.vleak = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        if self.usetaum:
            self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])

        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     
        
        self._epsilon = 1e-8
        
        self.sigmoid = nn.Sigmoid()
        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None

        nn.init.xavier_normal_(self.conv[0].weight.data)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))


    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self.cm.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
            # self.tau_m.data.clamp_(0,1)
        else:
            self.gleak.data.clamp_(0,1000)
        # self.tau_m.data = self._clip(self.tau_m.data)
    
    def forward(self, inputs, is_train):
        '''
        :param inputs: (B, C_in, S, H, W)
        :param hidden_state: (hx: (B, C, S, H, W), cx: (B, C, S, H, W))
        :return: (B, C_out, H, W)
        '''
        B, S, C, H, W = inputs.size() # 1 6 10 h w
        outputs = []
        cm_t = self.cm 
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):     

            wih = self.conv(inputs[:, t]) # wi*sig(x)+wh*sig(vpre)
            if self.tau_input:
                if self.usetaum:
                    numerator = (
                        self.tau_m * v_pre / (self.vleak + self.cm*self.sigmoid(wih)) + wih*self.E_revin                       
                    )
                    denominator = 1
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih*self.E_revin
                    )
                    denominator = cm_t + self.gleak + wih

            else:
                if self.usetaum:

                    numerator = (
                        self.tau_m * v_pre + wih# *self.E_revin
                    )
                    denominator = 1
    
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih
                    )
                    denominator = cm_t + self.gleak


            v_pre = numerator / (denominator + self._epsilon)

            v_pre = self.sigmoid(v_pre)
            
            outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:   
            return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        else:
            return outputs[-1]
   

class ContinuousFullyConnected(nn.Module):
    def __init__(self, hparams):
        super(ContinuousFullyConnected, self).__init__()
        # self.num_plane = hparams['num_plane']
        if hparams['ltcv1']:
            self._LTC_Conv = ConvLTC_v1(hparams).cuda()
        else:
            self._LTC_Conv = ConvLTC(hparams).cuda()

    def forward(self, events_fifo, is_train):
        projection = self._LTC_Conv(events_fifo, is_train)
        return projection
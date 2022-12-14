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
    # def __init__(self, config, in_channels, num_features, h_w, kernel_size=3, padding=1, stride=1, ode_unfolds=1, fixsm=True):
    # def __init__(self, in_channels, num_features, h_w, kernel_size, stride, padding=1, ode_unfolds=1):
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
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

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(num_features,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     
        
        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
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
#        self.cm.data = self._clip(self.cm.data)
#        self.gleak.data = self._clip(self.gleak.data)
        self.cm.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
        else:
            self.gleak.data.clamp_(0,1000)
        # self.tau_m.data = self._clip(self.tau_m.data)
    
    def forward(self, inputs, is_train):
        '''
        :param inputs: (B, C_in, S, H, W)
        :param hidden_state: (hx: (B, C, S, H, W), cx: (B, C, S, H, W))
        :return: (B, C_out, H, W)
        '''
        # gleak = self.cm/self.tau_m
        # gleak.retain_grad()
        # print("cm vleak gleak erevin is_leaf",self.cm.is_leaf, self.vleak.is_leaf,self.gleak.is_leaf, self.E_revin.is_leaf) 
        
        # print("gleak,vleak,cm",self.gleak.data,self.vleak.data,self.cm.data)
        #self.apply_weight_constraints()
        B, C, S, H, W = inputs.size()
        if self.in_channels == 5:
            S = S//5
        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm 
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):     

            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            if self.in_channels == 5:
                wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            else:
                wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)

            # denominator = self.cm_t + self.gleak 
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

            # v_pre = self.tanh(v_pre)
            v_pre = self.sigmoid(v_pre)
            # v_pre = self.tanh(v_pre)
            # v_pre.retain_grad()
            
            outputs.append(v_pre)
            # outputs.append(torch.tanh(v_pre))
            # outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:   
            return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        else:
            return outputs[-1]

class ConvLTC_v1(nn.Module):
    '''more general discrete form of LTC'''  
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
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

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(num_features,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     
        
        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
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
#        self.cm.data = self._clip(self.cm.data)
#        self.gleak.data = self._clip(self.gleak.data)
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
        # gleak = self.cm/self.tau_m
        # gleak.retain_grad()
        # print("cm vleak gleak erevin is_leaf",self.cm.is_leaf, self.vleak.is_leaf,self.gleak.is_leaf, self.E_revin.is_leaf) 
        
        # print("gleak,vleak,cm",self.gleak.data,self.vleak.data,self.cm.data)
        #self.apply_weight_constraints()
        B, C, S, H, W = inputs.size()
        if self.in_channels == 5:
            S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm 
       # if is_train:
        #    cm_t.retain_grad()
        v_pre = torch.zeros(B, self.num_features, H, W).cuda()
        for t in range(S-1,-1,-1):     

            # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            if self.in_channels == 5:
                wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            else:
                wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)

            # denominator = self.cm_t + self.gleak 
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
                        # self.tau_m * (v_pre + wih)# *self.E_revin
                    )
                    denominator = 1
                    # denominator = 1 + self.tau_m
    
                else:
                    numerator = (
                    cm_t * v_pre
                    + self.gleak * self.vleak
                    + wih
                    )
                    denominator = cm_t + self.gleak


            v_pre = numerator / (denominator + self._epsilon)

            # v_pre = self.tanh(v_pre)
            v_pre = self.sigmoid(v_pre)
            # v_pre = self.tanh(v_pre)
            # v_pre.retain_grad()
            
            outputs.append(v_pre)
            # outputs.append(torch.tanh(v_pre))
            # outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:   
            return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        else:
            return outputs[-1]
   

class ContinuousFullyConnected(nn.Module):
    def __init__(self, hparams):
        super(ContinuousFullyConnected, self).__init__()
        # self._continuous_fully_connected = \
        #  continuous_fully_connected.ContinuousFullyConnected(
        #     number_of_features, number_of_hidden_layers)
        self.num_plane = hparams['num_plane']
        if hparams['ltcv1']:
            self._LTC_Conv = ConvLTC_v1(hparams).cuda()
        else:
            self._LTC_Conv = ConvLTC(hparams).cuda()
        # self.SEblock = network_blocks.SELayer(hparams['nltc'],3)
        # self.eca_block = network_blocks.eca_block(hparams['nltc'])

    def forward(self, events_fifo, is_train):
        """Returns events projection.

        Args:
            events_fifo: first-in, first-out events queue of size
                        (batch_size, 2, number_of_events, height, width).

        Returns:
            projection: events projection of size (batch_size,
                        number_of_features, height, width).
        """
        # Note that polarity comes after timestamp in the fifo.
        # (events_timestamps,
        #  events_polarity) = (events_fifo[:, 0, ...].unsqueeze(1),
        #                    events_fifo[:, 1, ...].unsqueeze(1))
        events_polarity = events_fifo.clone()
        
        # Performance of three-planes seems not good.
        # neg_plane = events_polarity.clone()
        # pos_plane = events_polarity.clone()
        # neg_plane[neg_plane<0] = 0
        # pos_plane[pos_plane>0] = 0
        # three_plane_events_polarity = torch.cat((events_polarity, neg_plane, pos_plane),1)
        if self.num_plane == 1:       
            single_plane_events_polarity = events_polarity.clone()
            projection = self._LTC_Conv(single_plane_events_polarity, is_train)
        elif self.num_plane == 5:       
            single_plane_events_polarity = events_polarity.clone()
            projection = self._LTC_Conv(single_plane_events_polarity, is_train)
        else:
        # Two-planes
            neg_plane = events_polarity.clone()
            pos_plane = events_polarity.clone()
            neg_plane[neg_plane<0] = 0
            pos_plane[pos_plane>0] = 0
            two_plane_events_polarity = torch.cat((neg_plane, pos_plane),1)
            projection = self._LTC_Conv(two_plane_events_polarity, is_train)       
       
        # Single-plane
        

        # print("three_plane_events_polarity.size()",three_plane_events_polarity.size()) 1 3 10 320 384
        # projection = self._LTC_Conv(three_plane_events_polarity)
           # projection = self._LTC_Conv(two_plane_events_polarity, is_train)
#        projection = self._LTC_Conv(single_plane_events_polarity, is_train)
        # projection = self._continuous_fully_connected(events_timestamps,
        #                                               events_polarity)
        #projection = functional.leaky_relu(projection,inplace=False)
        # projection = torch.tanh(projection)
        # projection = self.SEblock(projection) 
        # projection = self.eca_block(projection)
        return projection


class TemporalConvolutional(nn.Module):
    def __init__(self, number_of_features=64):
        super(TemporalConvolutional, self).__init__()
        projection_modules = [
            convolution_3x1x1_with_relu(2, number_of_features)
        ] 
        projection_modules += [
            convolution_3x1x1_with_relu(number_of_features, number_of_features)
            for _ in range(2)
        ]
        self._projection_modules = nn.ModuleList(projection_modules)

    def forward(self, events_fifo):
        """Returns events projection.

        Args:
            events_fifo: first-in, first-out events queue of size
                        (batch_size, 2, 7, height, width).

        Returns:
            projection: events projection of size (batch_size,
                        number_of_features, height, width).
        """
        projection = events_fifo
        for projection_module in self._projection_modules:
            projection = projection_module(projection)
        return projection.max(dim=2)[0]



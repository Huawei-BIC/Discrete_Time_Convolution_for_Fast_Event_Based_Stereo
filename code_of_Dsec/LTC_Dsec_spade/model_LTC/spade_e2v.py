import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                if 'Float' in input_.type():
                    self.zero_tensors[state_size] = (
                        torch.zeros(state_size).to(input_.device).float(),
                        torch.zeros(state_size).to(input_.device).float()
                    )
                else:
                    self.zero_tensors[state_size] = (
                        torch.zeros(state_size).to(input_.device).half(),
                        torch.zeros(state_size).to(input_.device).half()
                    )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super(RecurrentConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.recurrent_block = ConvLSTM(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):

        x = self.relu(self.bn(self.conv0(x)))
        state = self.recurrent_block(x, prev_state)
        x = state[0]
        return x, state


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)
        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        # instance normalization
        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc)

        self.nhidden = nhidden
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.relu = nn.ReLU()

        # print('#########sdf####',norm_nc, label_nc, nhidden)
        self.conv2 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=1,dilation=1)
        self.conv3 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=2,dilation=2)
        self.conv4 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=3,dilation=3)
        self.conv5 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=4,dilation=4)
        # self.conv2 = nn.Conv2d(nhidden,nhidden//2,kernel_size=3,padding=1,dilation=1)
        # self.conv3 = nn.Conv2d(nhidden,nhidden//2,kernel_size=3,padding=3,dilation=3)
        # self.conv4 = nn.Conv2d(nhidden,nhidden//2,kernel_size=3,padding=4,dilation=4)
        # self.conv5 = nn.Conv2d(nhidden,nhidden//2,kernel_size=3,padding=5,dilation=5)

        self.final_conv = nn.Conv2d(norm_nc,norm_nc,kernel_size=3,padding=1)



    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        # print("x,segmap",x.size(),segmap.size())
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[-2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        
        img_features_2 = self.conv2(actv)
        img_features_3 = self.conv3(actv)
        img_features_4 = self.conv4(actv)
        img_features_5 = self.conv5(actv)
        # print("img_features_2",img_features_2.size())
        # actv_gamma = torch.cat((img_features_2[:,0:self.nhidden//4],img_features_3[:,0:self.nhidden//4],img_features_4[:,0:self.nhidden//4],img_features_5[:,0:self.nhidden//4]),dim=1)
        # actv_beta = torch.cat((img_features_2[:,self.nhidden//4:],img_features_3[:,self.nhidden//4:],img_features_4[:,self.nhidden//4:],img_features_5[:,self.nhidden//4:]),dim=1)

        # gamma = self.mlp_gamma(actv_gamma)
        # beta = self.mlp_beta(actv_beta)
        
        actv = torch.cat((img_features_2,img_features_3,img_features_4,img_features_5),dim=1)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        # print("normalized,gamma,beta",normalized.size(),gamma.size(),beta.size())
        out = normalized * (1 + gamma) + beta
        out = self.relu(self.final_conv(out))

        return out


class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, nom=3, scale=2):
        super(UpConvLayer, self).__init__()

        self.in_plane = in_channels
        self.out_plane = out_channels
        self.scale = scale
        self.planes = self.out_plane * scale ** 2

        self.conv0 = nn.Conv2d(self.in_plane, self.planes, kernel_size=5, padding=2, bias=True)
        self.icnr(scale=scale)
        self.shuf = nn.PixelShuffle(self.scale)

        self.norm = SPADE(self.out_plane, nom, nhidden)
        self.activation = nn.ReLU()

    def icnr(self, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = self.conv0.weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        self.conv0.weight.data.copy_(k)

    def forward(self, x, x_org):

        x = self.shuf(self.conv0(x))
        x = self.activation(self.norm(x, x_org))
        return x


class UpConvLayer3(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, nom=3):
        super(UpConvLayer3, self).__init__()

        self.in_plane = in_channels
        self.out_plane = out_channels
        self.scale = scale
        self.planes = self.out_plane * scale ** 2

        self.conv0 = nn.Conv2d(self.in_plane, self.planes, kernel_size=3, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(self.out_plane, self.out_plane, kernel_size=3, padding=1, bias=False)
        self.icnr(scale=scale)
        self.shuf = nn.PixelShuffle(self.scale)

        self.norm = SPADE(self.out_plane, nom)
        self.activation = nn.ReLU()

    def icnr(self, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = self.conv0.weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        self.conv0.weight.data.copy_(k)

    def forward(self, x, x_org):

        x = self.shuf(self.conv0(x))
        x = self.norm(x, x_org)
        x = self.activation(x)

        return x


class ConvLTC_v1(nn.Module):
    '''more general discrete form of LTC'''  
    def __init__(self, hparams, kernel_size=3, stride=1, padding=1, ode_unfolds=1):
        super().__init__()
       # torch.manual_seed(0)
       # torch.cuda.manual_seed(0)
        # in_channels, num_features, tau_input, taum_ini, usetaum, stream_opt, self.burn_in_time = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum'], hparams['stream_opt'], hparams['burn_in_time']
        in_channels, num_features, tau_input, taum_ini, usetaum = hparams['num_plane'], hparams['nltc'], hparams['use_erevin'], hparams['taum_ini'], hparams['usetaum']
        self.use_relu = hparams['use_relu']
        self.use_ltcsig = hparams['use_ltcsig']
        self.use_vtaum = hparams['use_vtaum']
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv = self._make_layer(in_channels, num_features, kernel_size, padding, stride)
        self.usetaum = usetaum    
        # self.stream_opt = stream_opt
        self.cm = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        self.vleak = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)
        if self.usetaum:
            if self.use_vtaum:
                self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,num_features,1,1)+taum_ini[1])
            else:
                self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])
            # self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,320,384)+taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1])*torch.rand(num_features,1,1)+taum_ini[1])

        if self.use_ltcsig:
            self.mu = nn.Parameter(0.1*torch.randn(num_features,1,1)+0)
            self.sigma = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)

        #self.tau_m = nn.Parameter((1.-5.)*torch.rand(num_features,1,1)+5.)
        self.E_revin = nn.Parameter(0.1*torch.randn(num_features,1,1)+1.0)# mean=1.0,std=0.1     

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU()

        #self._ode_unfolds = torch.Tensor(ode_unfolds).cuda()
        self._epsilon = 1e-8
        
        self.sigmoid = nn.Sigmoid()


        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None
        self.debug1 = []
        self.debug2 = []
        self.debug3 = []

        nn.init.xavier_normal_(self.conv[0].weight.data)


    def ltc_sigmoid(self, v_pre, mu, sigma):
        mues = v_pre - mu
        x = sigma * mues
        return self.sigmoid(x)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def apply_weight_constraints(self):
    #        self.cm.data = self._clip(self.cm.data)
    #        self.gleak.data = self._clip(self.gleak.data)
        self.cm.data.clamp_(0,1000)
        self.vleak.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
            # self.tau_m.data.clamp_(0,1)
        else:
            self.gleak.data.clamp_(0,1000)
        # self.tau_m.data = self._clip(self.tau_m.data)
    
    def forward(self, inputs, v_pre=None):
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
        if len(inputs.size()) == 4: # for bug in stream test
            B, C, H, W = inputs.size()
            # B= 1
            # inputs = torch.unsqueeze(inputs,0)
        else:
            B, C, S, H, W = inputs.size()
            # print('bcshw ', B,C,S,H,W)
        # if self.in_channels == 5:
            # S = S//5

        # v_pre = nn.Parameter(torch.zeros(B, self.num_features, H, W)).cuda()
        outputs = []
        # print("input.size()",inputs.size()) # 1 2 10 h w
       # cm_t = self.cm / (1. / self._ode_unfolds)
        cm_t = self.cm 
       # if is_train:
        #    cm_t.retain_grad()
        if v_pre == None:
            v_pre = torch.zeros(B, self.num_features, H, W).cuda()

        # S = 1            
        # for t in range(S-1,-1,-1):     
        # wih = self.conv(self.sigmoid(inputs[:, :,t])) # wi*sig(x)+wh*sig(vpre)
        # wih = self.conv(inputs[:, :,t]) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
            # wih = self.conv(inputs[:, 0,int(t*5):int((t+1)*5)]) # wi*sig(x)+wh*sig(vpre)
        wih = self.conv(inputs) # wi*sig(x)+wh*sig(vpre)
    

        if self.use_relu==1:
            wih = self.relu(wih)
        elif self.use_relu==2:
            wih = self.lrelu(wih)
        elif self.use_relu==3:
            wih = self.sigmoid(wih)
        # denominator = self.cm_t + self.gleak 
        if self.tau_input:
            if self.usetaum:
                numerator = (
                    self.tau_m * v_pre / (self.vleak + self.cm*self.sigmoid(wih)) + wih*self.E_revin # ltcv3                      
                    # (self.tau_m + self.cm*(self.sigmoid(wih)-.5)) * v_pre + wih*self.E_revin  # ltcv4                     
                    # (self.tau_m * self.cm* self.sigmoid(wih)) * v_pre + wih*self.E_revin # ltcv5                       
                    # self.tau_m * v_pre / (self.vleak + self.sigmoid(wih)) + wih*self.E_revin                       
                )
                denominator = 1
                # print(S, t)
                # self.debug1.append(self.tau_m / (self.vleak + self.cm*self.sigmoid(wih)))
                # self.debug2.append(v_pre)
                # self.debug3.append(wih)
            # self.debugv_pre
            else:
                numerator = (
                cm_t * v_pre
                + self.gleak * self.vleak
                + wih*self.E_revin
                )
                denominator = cm_t + self.gleak + wih

        else:
            if self.usetaum:
                if self.use_vtaum:
                    numerator = (
                        torch.sum(self.tau_m * v_pre.unsqueeze(2),1) + wih# cc11*bc1hw cc*c1
                        # self.tau_m * (v_pre + wih)# *self.E_revin
                    )
                else:
                    numerator = (self.tau_m * v_pre + wih)

                denominator = 1
                # denominator = 1 + self.tau_m
                # self.debug1.append(self.tau_m / (self.vleak + self.cm*self.sigmoid(wih)))
                # self.debug2.append(v_pre)
                # self.debug3.append(wih)
            else:
                numerator = (
                cm_t * v_pre
                + self.gleak * self.vleak
                + wih
                )
                denominator = cm_t + self.gleak


        v_pre = numerator / (denominator + self._epsilon) # [b c h w]

        # v_pre = self.tanh(v_pre)
        if self.use_ltcsig:
            v_pre = self.ltc_sigmoid(v_pre, self.mu, self.sigma)
        else:
            v_pre = self.sigmoid(v_pre)
     
        outputs.append(v_pre)

        self.debug = outputs[-1]
        # self.debug1 = torch.cat(self.debug1, 0)
        # self.debug2 = torch.cat(self.debug2, 0)
        # if self.stream_opt:   
        #     # return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        #     return torch.cat(outputs, 0).reshape(S, -1, H, W)[self.burn_in_time:] # only work for B=1
        # else:
        return outputs[-1]


class eca_block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Unet6(nn.Module):

    def __init__(self, hparams=None):
        super(Unet6, self).__init__()

        # self.use_ltc = hparams['use_ltc_spade']
        # self.use_conv2d = hparams['use_conv2d']
        # if self.use_ltc:
        #     if self.use_conv2d:
        #         self.fc = ConvLTC_v1(hparams, kernel_size=hparams['klsize'], stride=1, padding=hparams['pdsize'])
        #         self.fc_1 = nn.Conv2d(32, 32, 5, padding=2)
        #     else:
        #         self.fc = ConvLTC_v1(hparams, kernel_size=hparams['klsize'], stride=1, padding=hparams['pdsize'])
        # else:
        self.fc = nn.Conv2d(hparams['nltc'], 32, 5, padding=2)
        # layer 1
        self.rec0 = RecurrentConvLayer(32, 64, stride=1)
        self.rec1 = RecurrentConvLayer(64, 128, stride=2)
        self.rec2 = RecurrentConvLayer(128, 256, stride=2)
        # self.att1 = eca_block(256)
        # layer 2
        self.res0 = ResidualBlock(256, 256)
        self.res1 = ResidualBlock(256, 256)
        # self.att2 = eca_block(256)
        # layer 3
        self.up0 = UpConvLayer3(256, 128, nom=3)
        self.up1 = UpConvLayer3(128, 64, nom=3)
        self.up2 = RecurrentConvLayer(64, 32, stride=1)
        # self.att3 = eca_block(32)

        self.conv_img = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.bn_img = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prev_states, pred, v_pre=None):

        if prev_states is None:
            prev_states = [None] * 4

        x_org = pred
        # if self.use_ltc:
        #     # head = self.fc(x, v_pre)
        #     if self.use_conv2d:
        #         head_ctc, v_pre = self.fc(x, v_pre)
        #         head = self.relu(self.fc_1(head_ctc))
        #     else:
        #         head_ctc, v_pre = self.fc(x, v_pre)
        #         head = head_ctc    
        # else:
        head = self.relu(self.fc(x))
        # ------------
        x0, state0 = self.rec0(head, prev_states[0])
        x1, state1 = self.rec1(x0, prev_states[1])
        x2, state2 = self.rec2(x1, prev_states[2])
        # x2 = self.att1(x2) 
        # ------------
        x = self.res0(x2)
        x = self.res1(x)
        # x = self.att2(x)
        # ------------
        x = self.up0(x + x2, x_org)
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            x1 = F.interpolate(x1,(x.size(2),x.size(3)),mode='nearest')
        x = self.up1(x + x1, x_org)
        if x.size(2) != x0.size(2) or x.size(3) != x0.size(3):
            x0 = F.interpolate(x0,(x.size(2),x.size(3)),mode='nearest')
        x, state3 = self.up2(x + x0, prev_states[3])
        # x = self.att3(x)
        img_feature_0 = x # take img_feature input as output of lstm
        # print('ige sum ', img_feature_0.sum())
        # ------------

        stats = [state0, state1, state2, state3]
        # prediction layer and activation
        if x.size(2) != head.size(2) or x.size(3) != head.size(3):
            x = F.interpolate(x,(head.size(2),head.size(3)),mode='nearest')
        x = self.conv_img(self.relu(x + head))
        x = self.sigmoid(self.bn_img(x))
        # print('ige sum after', img_feature_0.sum())

        return x, stats, v_pre, img_feature_0






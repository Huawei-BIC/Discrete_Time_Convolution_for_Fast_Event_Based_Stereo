# CopyriVght 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch import nn
from model_LTC import temporal_aggregation
from model_LTC.spade_e2v import SPADE
import torch.nn.functional as F
from model_LTC import embedding
from model_LTC import estimator
from model_LTC import matching
from model_LTC import network_pds as network
from model_LTC import network_blocks
from model_LTC import regularization
from model_LTC import size_adapter
from model_LTC import stay_embedding
import time
import numpy as np
class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class DenseDeepEventStereo(network.PdsNetwork):
    """Dense deep stereo network.

    The network is based on "Practical Deeps Stereo: Toward
    applications-friendly deep stereo matching" by Stepan Tulyakov et al.
    Compare to the parent, this network has additional
    temporal aggregation module that embedds local events sequence in
    every location.
    """
    def __init__(self, size_adapter_module, temporal_aggregation_module,
                 spatial_aggregation_module, matching_module,
                 regularization_module, estimator_module):
        super(DenseDeepEventStereo,
              self).__init__(size_adapter_module, spatial_aggregation_module,
                             matching_module, regularization_module,
                             estimator_module)
        self._temporal_aggregation = temporal_aggregation_module
        # self.sbt_embedding = embedding.Embedding(number_of_input_features=5,number_of_embedding_features=32)
        # self.stay_embedding = stay_embedding.Embedding(number_of_input_features=5, number_of_embedding_features=32)
        # self.conv1 = network_blocks.convolutional_block_3x3(5,16)
        # self.conv2 = network_blocks.convolutional_block_3x3(16,32)
        self.sbt_size_adapter = size_adapter.SizeAdapter()
        self.spade = SPADE(32,5)
        self.time1 = []
        self.time2 = []
        self.time3 = []
        self.time4 = []
        self.time5 = []
        self.time6 = []
        self.counter = 95
        

    @staticmethod
    def default_with_temporal_convolutions(maximum_disparity=63):
        """Returns default network with temporal convolutions."""
        stereo_network = DenseDeepEventStereo(
            size_adapter_module=size_adapter.SizeAdapter(),
            temporal_aggregation_module=temporal_aggregation.
            TemporalConvolutional(),
            spatial_aggregation_module=embedding.Embedding(
                number_of_input_features=64),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(), maximum_disparity=0),
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        stereo_network.set_maximum_disparity(maximum_disparity)
        return stereo_network

    @staticmethod
    def default_with_continuous_fully_connected(hyper_params, maximum_disparity=63):
        """Returns default network with continuous fully connected."""
        stereo_network = DenseDeepEventStereo(
            size_adapter_module=size_adapter.SizeAdapter(),
            temporal_aggregation_module=temporal_aggregation.
            ContinuousFullyConnected(hyper_params),
            spatial_aggregation_module=embedding.Embedding(
                number_of_input_features=hyper_params['nltc'], number_of_embedding_features=32,
                number_of_shortcut_features=8),
            # spatial_aggregation_module=embedding.Embedding(
            #     number_of_input_features=10),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(number_of_concatenated_descriptor_features=64, number_of_features=64, number_of_compact_matching_signature_features=8, number_of_residual_blocks=2), maximum_disparity=0),
            # matching_module=matching.Matching(
            #       operation=matching.MatchingOperation(), maximum_disparity=0)
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        stereo_network.set_maximum_disparity(maximum_disparity)
        return stereo_network

    def default_with_hand_crafted(maximum_disparity=63):
        """Returns default network with continuous fully connected."""
        stereo_network = DenseDeepEventStereo(
            size_adapter_module=size_adapter.SizeAdapter(),
            temporal_aggregation_module=Dummy(),
            spatial_aggregation_module=embedding.Embedding(
                number_of_input_features=4),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(), maximum_disparity=0),
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        stereo_network.set_maximum_disparity(maximum_disparity)
        return stereo_network

    # def forward(self, left_event_queue, right_event_queue):
    def forward(self,batch):
        """Returns sub-pixel disparity (or similarities in training mode).

        Args:
            left_event_queue: first-in, first-out queue for the left camera.
                              events of size (batch_size,
                              number_of_events_features=2,
                              number_of_events=7, height, width). The queue
                              contains timestamp and polaritiy of recentest
                              events in every location.
            right_event_queue: first-in, first-out queue for the right camera.

        Returns:
            disparity tensor of size (batch_size, height, width) in evaluation
            mode and similiarities of size (batch_size,
            number_of_disparities / 2, height, width) in traininig mode.
        """
        # input:[2, 2, 1, 320, 384]
        LTC_start_time = time.time()
        # self.counter +=1
        # print("leftqueue",left_event_queue.size())
        left_event_queue = batch['left']['event_queue']
        right_event_queue = batch['right']['event_queue']
        
        # t_1 = time.time()
        # print('successfully',self.counter)
        left_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(left_event_queue), self.training)
        


        right_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(right_event_queue), self.training)
        
        #if self.counter > 100:
        #  self.time1.append(time.time()-t_1)
        #  print(np.mean(self.time1))
        # left_descriptor, shortcut_from_left = self._embedding(left_projected_events)
        # right_descriptor = self._embedding(right_projected_events)[0]
        
        
        # left_projected_events [1, 32, h, w]
        # print("LTC Duration:{:.4f}s".format(time.time()-LTC_start_time))
        ###########################################################
        #    embedding_module  fusion before spatial embedding    # 
        ###########################################################
        # h,w = left_event_queue.size()[-2:]
        # reshape_left_sbt = self.sbt_size_adapter.pad(left_event_queue.reshape(1,15,5,h,w)[:,0])
        # reshape_right_sbt = self.sbt_size_adapter.pad(right_event_queue.reshape(1,15,5,h,w)[:,0])
        # 
        # left_sbt = self.stay_embedding(reshape_left_sbt)[0]
        # right_sbt = self.stay_embedding(reshape_right_sbt)[0]
        # print("left_projected_events, left_sbt",left_projected_events.size(),left_sbt.size())
        # left_projected_events = self.spade(left_projected_events, left_sbt)
        # right_projected_events = self.spade(right_projected_events, right_sbt)
        # batch['feature_map'] = left_projected_events.clone()
        # 
        # left_fusion, shortcut_from_left = self._embedding(left_projected_events)
        # right_fusion = self._embedding(right_projected_events)[0]

        ########################################################### 
        ########################################################### 
        
        ###########################################################
        #         avg pooling fusion before spatial embedding    # 
        ###########################################################
        # h,w = left_event_queue.size()[-2:]
        # reshape_left_sbt = self.sbt_size_adapter.pad(left_event_queue.reshape(1,15,5,h,w)[:,0])
        # reshape_right_sbt = self.sbt_size_adapter.pad(right_event_queue.reshape(1,15,5,h,w)[:,0])
        # 

        ###########################################################
        #    embedding_module  fusion after spatial embedding    # 
        ###########################################################
        # h,w = left_event_queue.size()[-2:]
        # reshape_left_sbt = self.sbt_size_adapter.pad(left_event_queue.reshape(1,15,5,h,w)[:,0])
        # reshape_right_sbt = self.sbt_size_adapter.pad(right_event_queue.reshape(1,15,5,h,w)[:,0])
        # left_sbt = self.sbt_embedding(reshape_left_sbt)[0]
        # right_sbt = self.sbt_embedding(reshape_right_sbt)[0]
        # # print("##############?????????",left_descriptor.size(),left_sbt.size())
        # left_fusion = self.spade(left_descriptor, left_sbt)
        # right_fusion = self.spade(right_descriptor, right_sbt)
        # batch['feature_map'] = left_fusion.clone()

        ########################################################### 
        ########################################################### 
        
        ###########################################################
        #         avg pooling fusion after spatial embedding    # 
        ###########################################################
        # t_1 = time.time()
        left_descriptor, shortcut_from_left = self._embedding(left_projected_events)
        right_descriptor = self._embedding(right_projected_events)[0]
        #if self.counter > 100:
        #  self.time2.append(time.time()-t_1)
        #  print('_embedding',np.mean(self.time2))

        # batch['before_spade'] = left_descriptor.clone()
        # np.save('before_spadexxx.npy',left_descriptor.clone().detach().cpu().numpy())
        # print('xxx',left_descriptor.shape)
        
        h,w = left_event_queue.size()[-2:]
        reshape_left_sbt = left_event_queue.reshape(1,15,5,h,w)[:,0]
        reshape_right_sbt = right_event_queue.reshape(1,15,5,h,w)[:,0]

        # h,w = reshape_left_sbt.size()[-2:]
          
        # reshape_left_sbt = self.conv1(reshape_left_sbt)
        # reshape_right_sbt = self.conv1(reshape_right_sbt)
        # reshape_left_sbt = F.avg_pool2d(reshape_left_sbt, 2, stride=2)
        # reshape_right_sbt = F.avg_pool2d(reshape_right_sbt, 2, stride=2)
        # 
        # reshape_left_sbt = self.conv2(reshape_left_sbt)
        # reshape_right_sbt = self.conv2(reshape_right_sbt)
        # left_pooling = F.avg_pool2d(reshape_left_sbt, 2, stride=2)
        # right_pooling = F.avg_pool2d(reshape_right_sbt, 2, stride=2)

        # left_fusion = self.spade(left_descriptor, left_pooling)
        # right_fusion = self.spade(right_descriptor, right_pooling)
        # t_1 = time.time()
        left_fusion = self.spade(left_descriptor, reshape_left_sbt)
        right_fusion = self.spade(right_descriptor, reshape_right_sbt)
        # left_fusion = self.spade(left_fusion, reshape_left_sbt)
        # right_fusion = self.spade(right_fusion, reshape_right_sbt)
        
        

        #if self.counter > 100:
        #  self.time3.append(time.time()-t_1)
        #  print(np.mean(self.time3))
        
        
        batch['after_spade'] = left_fusion.clone()

        ########################################################### 
        ########################################################### 
        
        # network_output = self.pass_through_network(left_projected_events,
        #                                            right_projected_events)[0]
        
        # t_1 = time.time()
        matching_signatures = self._matching(left_fusion,right_fusion)
        #if self.counter > 100:
        #  self.time4.append(time.time()-t_1)
        #  print(np.mean(self.time4))
        
        # t_1 = time.time()
        network_output = self._regularization(matching_signatures, shortcut_from_left)
        print("dsf",network_output.size())
        #if self.counter > 100:
        #  self.time5.append(time.time()-t_1)
        #  print(np.mean(self.time5))
        expand_h,expand_w = left_projected_events.size()[-2:]
        
        if not self.training:
            start_time = time.time()
            network_output = self._estimator(network_output)
            if network_output.size()[-2:] != (expand_h,expand_w):
              # print("network_output",network_output.size())
                network_output = F.interpolate(network_output.unsqueeze(1),(expand_h,expand_w),mode='nearest').squeeze(1)
           # if self.counter > 100:
           #   self.time6.append(time.time()-start_time)
           #   print(np.mean(self.time6))
        
        # expand_h,expand_w = left_projected_events.size()[-2:]
        
        if network_output.size()[-2:] != (expand_h,expand_w):
            # print("network_output",network_output.size())
            network_output = F.interpolate(network_output,(expand_h,expand_w),mode='nearest')
            #print("Estimation Duration:{:.4f}s".format(time.time()-start_time))
        print("network_output.size()", network_output.size())
        
        # network output 2 32 320 384
        return self._size_adapter.unpad(network_output)


def _shallow_spatial_aggregation(number_of_input_features=64,
                                 number_of_features=64):
    # Instance norm in before embedding.
    modules_list = [
        nn.InstanceNorm2d(number_of_input_features),
        network_blocks.convolution_block_2D_with_relu_and_instance_norm(
            number_of_input_features=number_of_input_features,
            number_of_output_features=number_of_features,
            kernel_size=3,
            stride=1)
    ]
    modules_list += [
        network_blocks.convolution_block_2D_with_relu_and_instance_norm(
            number_of_input_features=number_of_features,
            number_of_output_features=number_of_features,
            kernel_size=3,
            stride=1) for _ in range(3)
    ]
    return nn.Sequential(*modules_list)


def _convolution_1x1(number_of_input_features, number_of_output_features):
    return nn.Conv2d(number_of_input_features,
                     number_of_output_features,
                     kernel_size=1,
                     padding=0)


def _convolution_1x1_with_relu(number_of_input_features,
                               number_of_output_features):
    return nn.Sequential(
        _convolution_1x1(number_of_input_features, number_of_output_features),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


def _shallow_matching_module():
    matching_operation = nn.Sequential(_convolution_1x1_with_relu(128, 128),
                                       _convolution_1x1(128, 1))
    return matching.Matching(operation=matching_operation, maximum_disparity=0)


class ShallowEventStereo(nn.Module):
    """Shallow network with small spatial context.

    The network is similar to
    "Computing the Stereo Matching Cost with a Convolutional Neural Network"
    by Jure Zbontar and Yann LeCun.

    The spatial embedding is performed by 4 sequential 3 x 3 convolutions with
    64 features and ReLU. The matching cost computation is performed by two
    fully connected layers with 128 features.
    """
    def __init__(self, temporal_aggregation_module, spatial_aggregation_module,
                 matching_module, estimator_module):
        super(ShallowEventStereo, self).__init__()
        self._temporal_aggregation = temporal_aggregation_module
        self._spatial_aggregation = spatial_aggregation_module
        self._matching = matching_module
        self._estimator = estimator_module

    @staticmethod
    def default_with_hand_crafted():
        """Returns default network with hand crafted temporal aggregation."""
        stereo_network = ShallowEventStereo(
            temporal_aggregation_module=Dummy(),
            spatial_aggregation_module=_shallow_spatial_aggregation(
                number_of_input_features=4),
            matching_module=_shallow_matching_module(),
            estimator_module=estimator.SubpixelMap(half_support_window=2,
                                                   disparity_step=1))
        stereo_network.set_maximum_disparity(maximum_disparity=38)
        return stereo_network

    @staticmethod
    def default_with_temporal_convolutions():
        """Returns default network with temporal convolutions."""
        stereo_network = ShallowEventStereo(
            temporal_aggregation_module=temporal_aggregation.
            TemporalConvolutional(),
            spatial_aggregation_module=_shallow_spatial_aggregation(),
            matching_module=_shallow_matching_module(),
            estimator_module=estimator.SubpixelMap(half_support_window=2,
                                                   disparity_step=1))
        stereo_network.set_maximum_disparity(maximum_disparity=38)
        return stereo_network

    @staticmethod
    def default_with_continuous_fully_connected():
        """Returns default network with temporal convolutions."""
        stereo_network = ShallowEventStereo(
            temporal_aggregation_module=temporal_aggregation.
            ContinuousFullyConnected(),
            spatial_aggregation_module=_shallow_spatial_aggregation(),
            matching_module=_shallow_matching_module(),
            estimator_module=estimator.SubpixelMap(half_support_window=2,
                                                   disparity_step=1))
        stereo_network.set_maximum_disparity(maximum_disparity=38)
        return stereo_network

    def set_maximum_disparity(self, maximum_disparity):
        """Reconfigure network for different disparity range."""
        self._matching.set_maximum_disparity(maximum_disparity - 1)

    def forward(self, left_event_queue, right_event_queue):
        left_embedding = self._spatial_aggregation(
            self._temporal_aggregation(left_event_queue))
        right_embedding = self._spatial_aggregation(
            self._temporal_aggregation(right_event_queue))
        matching_cost = self._matching(left_embedding,
                                       right_embedding).squeeze(dim=1)
        if not self.training:
            return self._estimator(matching_cost)
        return matching_cost


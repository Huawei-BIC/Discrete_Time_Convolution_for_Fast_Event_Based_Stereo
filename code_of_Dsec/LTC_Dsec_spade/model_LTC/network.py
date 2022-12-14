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
# from model_LTC import stay_embedding
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
    def __init__(self, size_adapter_module, temporal_aggregation_module,spade_module,
                 spatial_aggregation_module, matching_module,
                 regularization_module, estimator_module):
        super(DenseDeepEventStereo,
              self).__init__(size_adapter_module, spatial_aggregation_module,
                             matching_module, regularization_module,
                             estimator_module)
        self._temporal_aggregation = temporal_aggregation_module
        self.sbt_size_adapter = size_adapter.SizeAdapter()
        # self.spade = SPADE(32,10)
        self.spade = spade_module

    @staticmethod
    def default_with_continuous_fully_connected(hyper_params, maximum_disparity, embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks):
        """Returns default network with continuous fully connected."""
        stereo_network = DenseDeepEventStereo(
            size_adapter_module=size_adapter.SizeAdapter(),
            temporal_aggregation_module=temporal_aggregation.
            ContinuousFullyConnected(hyper_params),
            # spade_module=SPADE(hyper_params['nltc'],hyper_params['num_plane']),
            spade_module=SPADE(embedding_features,hyper_params['num_plane']),
            spatial_aggregation_module=embedding.Embedding(
                number_of_input_features=hyper_params['nltc'], number_of_embedding_features=embedding_features,
                number_of_shortcut_features=embedding_shortcuts),
            # spatial_aggregation_module=embedding.Embedding(
            #     number_of_input_features=10),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(number_of_concatenated_descriptor_features=matching_concat_features, number_of_features=matching_features, number_of_compact_matching_signature_features=matching_shortcuts, number_of_residual_blocks=matching_residual_blocks), maximum_disparity=0),
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        stereo_network.set_maximum_disparity(maximum_disparity)
        return stereo_network

    def forward(self,batch):
        # input:[2, 2, 1, 320, 384]
        LTC_start_time = time.time()
        left_event_queue = batch['left']['event_queue']
        right_event_queue = batch['right']['event_queue']
        
        left_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(left_event_queue), self.training)

        right_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(right_event_queue), self.training)
        
        left_descriptor, shortcut_from_left = self._embedding(left_projected_events)
        right_descriptor = self._embedding(right_projected_events)[0]
        reshape_left_sbt = left_event_queue[:,0]
        reshape_right_sbt = right_event_queue[:,0]
        left_fusion = self.spade(left_descriptor, reshape_left_sbt)
        right_fusion = self.spade(right_descriptor, reshape_right_sbt)
        
        batch['after_spade'] = left_fusion.clone()

        matching_signatures = self._matching(left_fusion,right_fusion)
        
        network_output = self._regularization(matching_signatures, shortcut_from_left)
        expand_h,expand_w = left_projected_events.size()[-2:]
        
        if not self.training:
            start_time = time.time()
            network_output = self._estimator(network_output)
            if network_output.size()[-2:] != (expand_h,expand_w):
              # print("network_output",network_output.size())
                network_output = F.interpolate(network_output.unsqueeze(1),(expand_h,expand_w),mode='nearest').squeeze(1)
        
        if network_output.size()[-2:] != (expand_h,expand_w):
            network_output = F.interpolate(network_output,(expand_h,expand_w),mode='nearest')
        # network output 2 32 320 384
        return self._size_adapter.unpad(network_output)




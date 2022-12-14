# CopyriVght 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch import nn
import torch
from model_LTC import temporal_aggregation

from model_LTC import embedding
from model_LTC import estimator
from model_LTC import matching
from model_LTC import network_pds as network
from model_LTC import network_blocks
from model_LTC import regularization
from model_LTC import size_adapter

import time

class DenseDeepEventStereo(network.PdsNetwork):
    def __init__(self, size_adapter_module, temporal_aggregation_module,
                 spatial_aggregation_module, matching_module,
                 regularization_module, estimator_module):
        super(DenseDeepEventStereo,
              self).__init__(size_adapter_module, spatial_aggregation_module,
                             matching_module, regularization_module,
                             estimator_module)
        self._temporal_aggregation = temporal_aggregation_module

    @staticmethod
    def default_with_continuous_fully_connected(hyper_params, maximum_disparity, embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks):
        """Returns default network with continuous fully connected."""
        stereo_network = DenseDeepEventStereo(
            size_adapter_module=size_adapter.SizeAdapter(),
            temporal_aggregation_module=temporal_aggregation.
            ContinuousFullyConnected(hyper_params),
            spatial_aggregation_module=embedding.Embedding(
                number_of_input_features=hyper_params['nltc'], number_of_embedding_features=embedding_features,
                number_of_shortcut_features=embedding_shortcuts),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(number_of_concatenated_descriptor_features=matching_concat_features, number_of_features=matching_features, number_of_compact_matching_signature_features=matching_shortcuts, number_of_residual_blocks=matching_residual_blocks), maximum_disparity=0),
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        stereo_network.set_maximum_disparity(maximum_disparity)
        return stereo_network

    def forward(self, left_event_queue, right_event_queue):
        # input:[2, 2, 1, 320, 384]
        LTC_start_time = time.time()
        left_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(left_event_queue), self.training)
        # print("leftpro",left_projected_events.size()) 1 32 320 384
        
        right_projected_events = self._temporal_aggregation(
            self._size_adapter.pad(right_event_queue), self.training)
        network_output = self.pass_through_network(left_projected_events,
                                                   right_projected_events)[0]
        if not self.training:
            start_time = time.time()
            network_output = self._estimator(network_output)
        return self._size_adapter.unpad(network_output)



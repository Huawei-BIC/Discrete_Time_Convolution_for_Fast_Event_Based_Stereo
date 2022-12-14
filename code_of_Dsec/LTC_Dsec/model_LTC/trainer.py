# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import torch as th
# import torch.distributed as dist
from torch import optim
import cv2
import yaml
import numpy as np
# from model_LTC import dataset_constants
from model_LTC import network

from model_LTC import errors
from model_LTC import pds_trainer
from model_LTC import trainer_pds as trainer
from model_LTC import visualization


class Trainer(pds_trainer.PdsTrainer):
    
    def _initialize_filenames(self):
        super(Trainer, self)._initialize_filenames()
        self._left_events_template = os.path.join(
            self._experiment_folder, 'example_{0:02d}_events.png')
        self._nonmasked_estimated_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:02d}_disparity_nomask_epoch_{1:03d}.png')

    def _run_network(self, batch_or_example):
        # print("batch_or_example",batch_or_example)
        batch_or_example['network_output'] = self._network(
            batch_or_example['left']['event_queue'],
            batch_or_example['right']['event_queue'])

    def _compute_error(self, example):
        
        estimated_disparity_all = example['network_output']
        ground_truth_disparity_all = example['left']['disparity_image']

        one = list()
        two = list()
        mean = list()
        root = list()

        for i in range(len(example['left']['disparity_image'])):

            ground_truth_disparity = ground_truth_disparity_all[i] 
            estimated_disparity = estimated_disparity_all[i]
            test_mask = (ground_truth_disparity == float('inf'))
            test_gtd = ground_truth_disparity[~test_mask].clone().detach()
            original_dataset = self._test_set_loader.dataset
            binary_error_map, one_pixel_error = errors.compute_n_pixels_error(
                estimated_disparity, ground_truth_disparity, n=1.0)
            one.append(one_pixel_error)
            _, two_pixel_error = errors.compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=2.0)
            two.append(two_pixel_error)
            mean_disparity_error = errors.compute_absolute_error(
                estimated_disparity, ground_truth_disparity)[1]
            mean.append(mean_disparity_error)
            RMSE = errors.RMSE(estimated_disparity, ground_truth_disparity)[1] 
            root.append(RMSE)
        example['error'] = {
            'one_pixel_error': np.mean(one),
            'two_pixel_error': np.mean(two),
            'mean_average_error': np.mean(mean),
            'root_mean_square_error': np.mean(root)
        }

    def _report_training_progress(self):
        """Plot and print training loss and validation error every epoch."""
        test_errors = list(
            map(lambda element: element['one_pixel_error'], self._test_errors))
        visualization.plot_losses_and_errors(self._plot_filename,
                                             self._training_losses,
                                             test_errors)
        self._logger.log('epoch {0:02d} ({1:02d}) : '
                         'training loss = {2:.5f}, '
                         '1PE = {3:.2f} [%], '
                         '2PE = {4:.2f} [%], '
                         'mean average error = {5:.3f} [pix], '
                         'root mean square error = {6:.3f}, '
                         'learning rate = {7:.5f}.'.format(
                             self._current_epoch + 1, self._end_epoch,
                             self._training_losses[-1],
                             self._test_errors[-1]['one_pixel_error'],
                             self._test_errors[-1]['two_pixel_error'],
                             self._test_errors[-1]['mean_average_error'],
                             self._test_errors[-1]['root_mean_square_error'],
                             trainer.get_learning_rate(self._optimizer)))

    def _report_test_results(self, error, time):
        self._logger.log('Testing results: '
                         '1PE = {0:.3f} [%], '
                         '2PE = {1:.3f} [%], '
                         'mean average error = {2:.3f} [pix], '
                         'root mean square error = {3:.3f}, '
                         'time-per-image = {4:.2f} [sec].'.format(
                             error['one_pixel_error'],
                             error['two_pixel_error'],
                             error['mean_average_error'],
                             error['root_mean_square_error'], time))


def initialize_optimizer(stereo_network, main_lr, temporal_aggregation_lr):
    return optim.RMSprop(
        [{
            "params": stereo_network._temporal_aggregation.parameters(),
            "lr": temporal_aggregation_lr
        }, {
            "params": stereo_network._embedding.parameters()
        }, {
            "params": stereo_network._matching.parameters()
        }, {
            "params": stereo_network._regularization.parameters()
        }], main_lr)
    
            
def initialize_network(hyper_params, maximum_disparity,embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks):
    print("network",hyper_params, maximum_disparity,embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks)
    network_class = network.DenseDeepEventStereo
    return network_class.default_with_continuous_fully_connected(hyper_params,maximum_disparity,embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks)

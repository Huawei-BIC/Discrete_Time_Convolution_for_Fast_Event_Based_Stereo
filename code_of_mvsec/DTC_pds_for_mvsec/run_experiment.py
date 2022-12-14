#!/usr/bin/env python3
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#copied from run_experiment_ip.py, develop for running LTC network independently, 2021.9.15

import os
import numpy as np

from torch.optim import lr_scheduler
from torch.utils import data
import torch as th

from model_LTC import dataset
from model_LTC import trainer
from model_LTC import loss

import argparse
import time
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import random

# num_gpu = 3
# os.environ["CUDA_VISIBLE_DEVICES"]=str(num_gpu)

def _initialize_dataloaders(dataset_folder, split_number, queue_time_horizon,
                            temporal_aggregation_type, queue_capacity,
                            use_full_ground_truth, test_mode, debug_mode, data_hparams):
    print("Dataset preparation begins.")
    start_time = time.time()
    
    sets = dataset.IndoorFlying.split(dataset_folder,
                                      split_number, data_hparams, random_seed=0)
    training_set = sets[0]
    if test_mode:
        test_set = sets[2]
    else:
        test_set = sets[1]
    # dataset_transformers = trainer.initialize_transformers(
    #     temporal_aggregation_type, queue_capacity, use_full_ground_truth)
    # training_set.set_time_horizon(queue_time_horizon)
    # test_set.set_time_horizon(queue_time_horizon)
    # training_set._transformers = dataset_transformers
    # test_set._transformers = dataset_transformers
    
    # training_sampler = th.utils.data.distributed.DistributedSampler(training_set)
    
    training_set_loader = data.DataLoader(training_set,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          # sampler=training_sampler,
                                          pin_memory=True,
                                          num_workers=6)
    test_set_loader = data.DataLoader(test_set, batch_size=1,shuffle=False, pin_memory=True, num_workers=6)

    print("preparation ends. Duration:", time.time()-start_time)
    return training_set_loader, test_set_loader


def _initialize_parameters(dataset_folder, temporal_aggregation_type,
                           experiment_folder, split_number, queue_capacity,
                           queue_time_horizon, use_full_ground_truth,
                           test_mode, use_shallow_network, main_lr,
                           temporal_aggregation_lr, debug_mode, hyper_params, spec_title, data_hparams):
    #if dist.get_rank() == 0:
    print("preparing parameters.")
    start_time = time.time()
    training_set_loader, test_set_loader = _initialize_dataloaders(
        dataset_folder, split_number, queue_time_horizon,
        temporal_aggregation_type, queue_capacity, use_full_ground_truth,
        test_mode, debug_mode, data_hparams)
    stereo_network = trainer.initialize_network(temporal_aggregation_type,
                                                use_shallow_network, hyper_params)
    optimizer = trainer.initialize_optimizer(stereo_network, main_lr,
                                             temporal_aggregation_lr)
    learning_rate_scheduler = lr_scheduler.MultiStepLR(
        #optimizer, milestones=[8, 10, 12, 14, 16, 18, 20, 22], gamma=0.5)
        optimizer, milestones=[15, 23], gamma=0.5)
        # optimizer, milestones=[8, 10], gamma=0.5)
        #optimizer, milestones=[8], gamma=0.5)

        # optimizer, milestones=[8,12,16,20], gamma=0.5)
    if use_shallow_network:
        criterion = loss.SubpixelCrossEntropy(diversity=1.0, disparity_step=1)
    else:
        criterion = loss.SubpixelCrossEntropy(diversity=1.0, disparity_step=2)
    if th.cuda.is_available():
        criterion.cuda()
        stereo_network.cuda() 
    if debug_mode:
        end_epoch = 1 
    else:
        end_epoch = 44 #22
    # if dist.get_rank() == 0:
    print("Preparation ends. Duration: ", time.time()-start_time)
    print("Parameters:",np.sum([p.numel() for p in stereo_network.parameters()]).item())
    return {
        'network': stereo_network,
        'optimizer': optimizer,
        'criterion': criterion,
        'learning_rate_scheduler': learning_rate_scheduler,
        'training_set_loader': training_set_loader,
        'test_set_loader': test_set_loader,
        'end_epoch': end_epoch,
        'experiment_folder': experiment_folder,
        'spec_title': spec_title
    }

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--experiment_folder', default= 'experiments/', type=str)

parser.add_argument('--checkpoint_file', default= None, type=str)

parser.add_argument('--dataset_folder', default='mvsec_dataset', type=str)
parser.add_argument('--temporal_aggregation_type', choices=['hand_crafted', 'temporal_convolutional', 'continuous_fully_connected'], default='continuous_fully_connected')
parser.add_argument('--use_full_ground_truth', default=False, action='store_true')

parser.add_argument('--test_mode', default=False, action='store_true')
# parser.add_argument('--test_mode', default=True, action='store_true')

parser.add_argument('--split_number', choices=['1','3'],default='1')
parser.add_argument('--queue_time_horizon', type=float, default=None)
parser.add_argument('--queue_capacity', type=int, default=None)
parser.add_argument('--use_shallow_network', default=False, action='store_true')
parser.add_argument('--main_lr', type=float, default=None)
parser.add_argument('--temporal_aggregation_lr',type=float, default=None)
# parser.add_argument('--debug_mode', default=True, action='store_true')
parser.add_argument('--debug_mode', default=False, action='store_true')

parser.add_argument('--ltc_hparams', default={'use_erevin':False, 'num_plane':5, 'taum_ini':[.5,.8], 'nltc': 32, 'usetaum':True, 'ltcv1':True}, type=dict)
# parser.add_argument('--data_hparams', default={'use10ms': False, 'usenorm':False, 'pre_nframes':10}, type=dict)
parser.add_argument('--data_hparams', default={'use10ms': True, 'usenorm': False, 'pre_nframes':10}, type=dict)
parser.add_argument('--share_hparams', default={'stream_opt':False, 'burn_in_time':5}, type=dict)

parser.add_argument('--spec_title', default = 4000, type=int) # for normal training and general testing 
args = parser.parse_args()

# parser.add_argument('--checkpoint_file', default= args.experiment_folder+'010_checkpoint.bin', type=str)
args = parser.parse_args()


for i,v in args.share_hparams.items():
    args.ltc_hparams[i] = v
    args.data_hparams[i] = v


def set_seed(seed=0):
    random.seed(0)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


if __name__ == '__main__':

    set_seed(12345)

    (default_main_lr, default_temporal_aggregation_lr,
     default_queue_time_horizon,
     default_queue_capacity) = trainer.default_network_dependent_parameters(
         args.temporal_aggregation_type, args.use_shallow_network)

    if args.main_lr is None:
        main_lr = default_main_lr
    if args.temporal_aggregation_lr is None:
        temporal_aggregation_lr = default_temporal_aggregation_lr
    if args.queue_time_horizon is None:
        queue_time_horizon = default_queue_time_horizon
    if args.queue_capacity is None:
        queue_capacity = default_queue_capacity

    dataset_folder = args.dataset_folder
    experiment_folder = args.experiment_folder

    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

    parameters = _initialize_parameters(
        dataset_folder, args.temporal_aggregation_type, experiment_folder,
        int(args.split_number), queue_capacity, queue_time_horizon,
        args.use_full_ground_truth, args.test_mode, args.use_shallow_network, main_lr,
        temporal_aggregation_lr, args.debug_mode, args.ltc_hparams, args.spec_title, args.data_hparams)

    if args.temporal_aggregation_type == 'hand_crafted':
        stereo_trainer = trainer.TrainerForHandcrafted(parameters)
    else:
        stereo_trainer = trainer.Trainer(parameters)

    if args.checkpoint_file:
        stereo_trainer.load_checkpoint(args.checkpoint_file, load_only_network=True)

    if args.test_mode:
        # if dist.get_rank() == 0:
        print("Testing.")
        stereo_trainer.test()
        print('LTC cm ', stereo_trainer._network._temporal_aggregation._LTC_Conv.cm)
        #stereo_trainer._network.parameters
    else:
        # if dist.get_rank() == 0:
        print("Training.")
        stereo_trainer.train()































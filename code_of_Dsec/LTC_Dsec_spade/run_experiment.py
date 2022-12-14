import os
import numpy as np

from torch.optim import lr_scheduler
from torch.utils import data
import torch as th
from pathlib import Path
from model_LTC import trainer
from model_LTC import loss
from model_LTC import provider

import argparse
import time
import torch.distributed as dist
import random

def _initialize_dataloaders(dataset_folder, test_mode, data_hparams, pre_frame, all_data):
    print("Dataset preparation begins.")
    start_time = time.time()
    
    dataset_provider = provider.DatasetProvider(Path(dataset_folder))
    training_set, test_set = dataset_provider.get_train_and_valid_dataset(pre_frame, all_data)

    training_set_loader = data.DataLoader(training_set,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=6)
    test_set_loader = data.DataLoader(test_set, batch_size=4,shuffle=False, pin_memory=True, num_workers=6)

    print("preparation ends. Duration:", time.time()-start_time)
    return training_set_loader, test_set_loader


def _initialize_parameters(dataset_folder, experiment_folder,test_mode, main_lr,temporal_aggregation_lr, hyper_params, spec_title, data_hparams, DA, pre_frame, end_epoch,milestone, all_data, maximum_disparity,embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks ):
    print("preparing parameters.")
    start_time = time.time()
    training_set_loader, test_set_loader = _initialize_dataloaders(dataset_folder,test_mode, data_hparams, pre_frame, all_data)
    stereo_network = trainer.initialize_network(hyper_params,maximum_disparity,embedding_features, embedding_shortcuts,matching_concat_features,matching_features,matching_shortcuts,matching_residual_blocks)
    
    optimizer = trainer.initialize_optimizer(stereo_network, main_lr,
                                             temporal_aggregation_lr)
    learning_rate_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestone, gamma=0.5)
    criterion = loss.SubpixelCrossEntropy(diversity=1.0, disparity_step=2)
    criterion.cuda()
    stereo_network.cuda() 
    if th.cuda.device_count()>1:
        stereo_network = th.nn.DataParallel(stereo_network)
    
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
        'spec_title': spec_title,
        'DA': DA
    }


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--experiment_folder', default= None, type=str)
parser.add_argument('--checkpoint_file', default= None, type=str)
parser.add_argument('--dataset_folder', type=str)
parser.add_argument('--test_mode', default=False, action='store_true')
parser.add_argument('--main_lr', type=float, default=None)
parser.add_argument('--maximum_disparity', type=int, default=127)
parser.add_argument('--pre_frame', type=int, default=6)
parser.add_argument('--temporal_aggregation_lr',type=float, default=None)
parser.add_argument('--end_epoch',type=int, default=22)

parser.add_argument('--milestone',type=int, nargs='+')
parser.add_argument('--embedding_features', default=32, type=int)
parser.add_argument('--embedding_shortcuts', default=8, type=int)
parser.add_argument('--matching_concat_features', default=64, type=int)
parser.add_argument('--matching_features', default=64, type=int)
parser.add_argument('--matching_shortcuts', default=8, type=int)
parser.add_argument('--matching_residual_blocks', default=2, type=int)


parser.add_argument('--all_data',default=False,action='store_true')
parser.add_argument('--DA',default=False, action='store_true')
parser.add_argument('--ltc_hparams', default={'use_erevin':False, 'taum_ini':[.5,.8], 'nltc': 32, 'usetaum':True, 'ltcv1':True}, type=dict)
parser.add_argument('--num_plane',type=int, default=10)
parser.add_argument('--data_hparams', default={'use10ms': True, 'usenorm': False, 'pre_nframes':10}, type=dict)
parser.add_argument('--share_hparams', default={'stream_opt':False, 'burn_in_time':5}, type=dict)
parser.add_argument('--spec_title', default = 4000, type=int) # for normal training and general testing 
args = parser.parse_args()
args.ltc_hparams['num_plane'] = args.num_plane
# print(args.ltc_hparams)
print(args)


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

    main_lr = 1e-3
    temporal_aggregation_lr = 5e-3

    dataset_folder = args.dataset_folder
    experiment_folder = args.experiment_folder

    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)


    parameters = _initialize_parameters(
        dataset_folder, experiment_folder, args.test_mode, main_lr,temporal_aggregation_lr, args.ltc_hparams, args.spec_title, args.data_hparams, args.DA, args.pre_frame,args.end_epoch, args.milestone, args.all_data, args.maximum_disparity,args.embedding_features, args.embedding_shortcuts,args.matching_concat_features,args.matching_features,args.matching_shortcuts,args.matching_residual_blocks )

    stereo_trainer = trainer.Trainer(parameters)
    if args.checkpoint_file:
        stereo_trainer.load_checkpoint(args.checkpoint_file, load_only_network=True)

    if args.test_mode:
        print("Testing.")
        stereo_trainer.test()
        print('LTC cm ', stereo_trainer._network._temporal_aggregation._LTC_Conv.cm)
    else:
        print("Training.")
        stereo_trainer.train()





























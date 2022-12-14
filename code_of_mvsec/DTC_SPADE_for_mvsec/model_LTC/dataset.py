# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import PIL.Image
import numpy as np
import os
import random
# from functools import reduce
import torch
from model_LTC import dataset_constants
from model_LTC import transforms
# For test we use same frames as
# "Realtime Time Synchronized Event-based Stereo"
# by Alex Zhu et al. for consistency of test results.


FRAMES_FILTER_FOR_TEST = {
    'indoor_flying': {
        1: list(range(140, 1201)),
        2: list(range(120, 1421)),
        3: list(range(73, 1616)),
        4: list(range(190, 290))
    }
}


# For the training we use different frames, since we found
# that frames recomended by "Realtime Time Synchronized
# Event-based Stereo" by Alex Zhu include some still frames.
FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying': {
        1: list(range(80, 1260)),
        2: list(range(160, 1580)),
        3: list(range(125, 1815)),
        4: list(range(190, 290))
    }
}


def _filter_examples(examples, frames_filter):
    return [
        example for example in examples
        if frames_filter[example['experiment_name']][example['experiment_number']] is None or example['frame_index'] in
        frames_filter[example['experiment_name']][example['experiment_number']]
    ]


def _get_examples_from_experiments(experiments, dataset_folder):
    examples = []
    for experiment_name, experiment_numbers in experiments.items():
        for experiment_number in experiment_numbers:
            examples += _get_examples_from_experiment(experiment_name,
                                                      experiment_number,
                                                      dataset_folder)
    return examples


def _get_examples_from_experiment(experiment_name, experiment_number,
                                  dataset_folder):
    examples = []
    paths = dataset_constants.experiment_paths(experiment_name,
                                               experiment_number,
                                               dataset_folder)
    timestamps = np.loadtxt(paths['timestamps_file'])
    frames_number = timestamps.shape[0]

    for frame_index in range(frames_number):
        example = {}
        example['experiment_name'] = experiment_name
        example['experiment_number'] = experiment_number
        example['frame_index'] = frame_index
        example['timestamp'] = timestamps[frame_index]
        example['left_image_path'] = paths['cam0']['image_file'] % frame_index
        example['disparity_image_path'] = paths['disparity_file'] % frame_index
        examples.append(example)

    return examples


def _get_image(image_path):
    # Not all examples have images.
    if not os.path.isfile(image_path):
        return np.zeros(
            (dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH),
            dtype=np.uint8)
    return np.array(PIL.Image.open(image_path)).astype(np.uint8)


def _get_disparity_image(disparity_image_path):
    disparity_image = _get_image(disparity_image_path)
    invalid_disparity = (
        disparity_image == dataset_constants.INVALID_DISPARITY)
    disparity_image = (disparity_image /
                       dataset_constants.DISPARITY_MULTIPLIER)
    disparity_image[invalid_disparity] = float('inf')
    return disparity_image


def normalize_features_to_zero_mean_unit_std(example):
    left_event_queue = example['left']['event_queue']
    right_event_queue = example['right']['event_queue']
    mean_left = left_event_queue[0].mean()
    std_left = left_event_queue[0].std()
    left_event_queue = (left_event_queue - mean_left)/(std_left+1e-10)
    example['left']['event_queue'] = left_event_queue

    mean_right = right_event_queue[0].mean()
    std_right = right_event_queue[0].std()
    right_event_queue = (right_event_queue - mean_right)/(std_right+1e-10)
    example['right']['event_queue'] = right_event_queue

    return example


def normalize_polarity(example):
    return normalize_features_to_zero_mean_unit_std(example)

# from torchvision import transforms

class MvsecDataset():
    def __init__(self, examples, dataset_folder, data_hparams, transformers_list=None, is_test_valid = False):
        self._examples = examples
        self._transformers = transformers_list
        self._dataset_folder = dataset_folder
        self._events_time_horizon = dataset_constants.TIME_BETWEEN_EXAMPLES
        self._number_of_events = float('inf')
        self.data_params = data_hparams
        self.usenorm = data_hparams['usenorm']
        self.use10ms = data_hparams['use10ms']
        self.pre_nframes = data_hparams['pre_nframes']
        self.burn_in_time = data_hparams['burn_in_time']
        self.stream_opt = data_hparams['stream_opt']
        self.transform=transforms.Compose([
                                        # transforms.ToPILImage(),
                                        # transforms.RandomHorizontalFlip(p=0.3),
                                        # transforms.RandomVerticalFlip(),
                                        transforms.RandomCrop(200,280,0.5)
                                        # transforms.RandomCropResize(probability=0.5
                                        #      )
                                        # transforms.ToTensor()
                                        ])
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!is_test_valid',is_test_valid)
        self.is_test_valid = is_test_valid

    def set_time_horizon(self, time):
        self._events_time_horizon = time
        self._number_of_events = float('inf')

    def set_number_of_events(self, number_of_events):
        self._events_time_horizon = float('inf')
        self._number_of_events = number_of_events

    def _get_event_queue(self, experiment_number, frame_index):

        if self.use10ms:
            left_path = 'mvsec/'
            left_path += 'indoor_flying_{}'.format(experiment_number)+'/event0_10ms_frame/'
            right_path = 'mvsec/'
            right_path += 'indoor_flying_{}'.format(experiment_number)+'/event1_10ms_frame/'

        else:
            left_path = 'mvsec_dataset/'
            left_path += 'indoor_flying_{}'.format(experiment_number)+'/event_frame0_1/'
            right_path = 'mvsec_dataset/'
            right_path += 'indoor_flying_{}'.format(experiment_number)+'/event_frame1_1/'

            # left_path = '/media/HDD1/personal_files/lengluziwei/mvsec/'
            # left_path += 'indoor_flying_{}'.format(experiment_number)+'/event0_50ms_f/' # [260, 346]
            # right_path = '/media/HDD1/personal_files/lengluziwei/mvsec/'
            # right_path += 'indoor_flying_{}'.format(experiment_number)+'/event1_50ms_f/'
       
        first_index = int(max(-1, (frame_index-self.pre_nframes-self.burn_in_time)))
        left_eq, right_eq = [], []
        for previous_frame_index in range(frame_index, first_index, -1):
        # for previous_frame_index in range(frame_index, first_index, -1)[::-1]:
            left_events_filename = left_path + "{:06d}.npy".format(previous_frame_index)
            right_events_filename = right_path + "{:06d}.npy".format(previous_frame_index)
            left_event = torch.from_numpy(np.load(left_events_filename)).reshape(-1, 260, 346) 
            right_event = torch.from_numpy(np.load(right_events_filename)).reshape(-1, 260, 346) 

            # print('left:',.shape)
            # left_event = torch.flip(left_event,[0])
            # for i in range(left_event.shape[0]):
            #     left_event[i] = self.transform(left_event[i])

            # right_event = torch.flip(left_event,[0])
            # for i in range(right_event.shape[0]):
            #     right_event[i] = self.transform(right_event[i])

            left_eq.append(torch.flip(left_event,[0]))# add flip to make t10.0, t10.1 ...t1.0, t1.1..t1.4 be loaded as t1.0,t1.1...t10.0..t10.4 in ltc forward
            right_eq.append(torch.flip(right_event, [0]))
        left_event_queue, right_event_queue = torch.unsqueeze(torch.cat(left_eq, 0),0), torch.unsqueeze(torch.cat(right_eq,0),0)
        # print("left_event_queue.size()",left_event_queue.size()) 1 10 260 346
        return left_event_queue.float(), right_event_queue.float()


    def _get_disparity_image_queue(self, disparity_image_path, experiment_number, frame_index):
        if self.stream_opt:
      
            left_path = 'mvsec_dataset/'
            left_path += 'indoor_flying_{}'.format(experiment_number)+'/disparity_image/'        
       
            first_index = int(max(-1, (frame_index-self.pre_nframes)))
            left_eq = []
            for previous_frame_index in range(frame_index, first_index, -1)[::-1]:
            # for previous_frame_index in range(frame_index, first_index, -1)[::-1]:
                sub_image_filename = left_path + "{:06d}.png".format(previous_frame_index)
                sub_image = _get_image(sub_image_filename)

                invalid_disparity = (
                    sub_image == dataset_constants.INVALID_DISPARITY)
                sub_image = (sub_image /
                                   dataset_constants.DISPARITY_MULTIPLIER)
                sub_image[invalid_disparity] = float('inf')

                left_eq.append(sub_image)# 

            disparity_image = np.array(left_eq)
            # print('left dis imag shape ', disparity_image.shape) : 10,260,346

        else:    
            disparity_image = _get_image(disparity_image_path)
            invalid_disparity = (
                disparity_image == dataset_constants.INVALID_DISPARITY)
            disparity_image = (disparity_image /
                               dataset_constants.DISPARITY_MULTIPLIER)
            disparity_image[invalid_disparity] = float('inf')

        return disparity_image

    def split_into_two(self, first_subset_size):
        return (self.__class__(self._examples[:first_subset_size],
                               self._dataset_folder, self.data_params,
                               transformers_list=self._transformers,is_test_valid=True),
                self.__class__(self._examples[first_subset_size:],
                               self._dataset_folder, self.data_params,
                               transformers_list=self._transformers,is_test_valid=True))

    def shuffle(self, random_seed=0):
        """Shuffle examples in the dataset.

        By setting "random_seed", one can ensure that order will be the
        same across different runs. This is usefull for visualization of
        examples during the traininig.
        """
        random.seed(random_seed)
        random.shuffle(self._examples)

    def subsample(self, number_of_examples, random_seed=0):
        """Keeps "number_of_examples" examples in the dataset.

        By setting "random_seed", one can ensure that subset of examples
        will be same in a different runs. This method is usefull for
        debugging.
        """
        random.seed(random_seed)
        self._examples = random.sample(self._examples, number_of_examples)

    def __len__(self):
        return len(self._examples)

    def get_example(self, index):
        example = self._examples[index]
     
        left_event_queue, right_event_queue = self._get_event_queue(example['experiment_number'],example['frame_index'])  # 1(polarity), 10(event_number), h, w
        # print('frame index',example['frame_index'])

       


        # left_event_queue = transform(left_event_queue)
        # right_event_queue = transform(right_event_queue)
        gt_disparity = self._get_disparity_image_queue(example['disparity_image_path'], example['experiment_number'], example['frame_index'])

        # np_data = np.array(left_event_queue)
        # np.save('data.npy',np_data)

        # np_data = np.array(gt_disparity)
        # np.save('data_gt.npy',np_data) 

        # print('Successfully')
        # print(left_event_queue.shape)
        # print(gt_disparity.shape)
        sample = {}
        sample['left'], sample['right'], sample['disp'] = left_event_queue, right_event_queue, gt_disparity
        # print('Successfully!!!!!!!!!!!!!!!!!!!!!',self.is_test_valid)
        if not self.is_test_valid:
            sample = self.transform(sample)
        left_event_queue, right_event_queue, gt_disparity = sample['left'], sample['right'], sample['disp']
        # print('After')
        # print(left_event_queue.shape)
        # print(gt_disparity.shape)
        # gt_disparity = transform(gt_disparity)
        
        # if self._transformers is not None:
        #     for t in self._transformers:
        #         left_event_queue = t(left_event_queue)
        #         right_event_queue = t(right_event_queue)
        #     print('Successfully')
        
        return {
            'left': {
                'image':
                _get_image(example['left_image_path']),
                # 'event_sequence':
                # left_event_sequence,
                'event_queue':
                left_event_queue,
                'disparity_image':
                gt_disparity,
            },
            'right': {
                # 'event_sequence': right_event_sequence
                'event_queue':right_event_queue
            },
            'timestamp': example['timestamp'],
            'frame_index': example['frame_index']
        }

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.get_example(index)
        if self.usenorm:
            example = normalize_polarity(example)
        # if self._transformers is not None:
        #     for t in self._transformers:
        #         example = ToTensor(example)
        #     print('Successfully')

        # if self._transformers is not None:
            # for transformer in self._transformers:
            #     example = transformer(example)
            # print("example['image']", example['left']['image'].size())
            # [260, 346]
            # print("example['disparity']", example['left']['disparity_image'].size())
            # [260, 346]
            # print("example['event_sequence']", example['left']['event_queue'].size())
            # [2, 1, 260, 346]
        return example

    @staticmethod
    def disparity_to_depth(disparity_image):
        """Converts disparity to depth."""
        raise NotImplementedError('"disparity_to_depth" method should '
                                  'be implemented in a child class.')

    @classmethod
    def dataset(cls, dataset_folder, experiments, data_hparams, frames_filter=None, is_test_valid=False):
        # print('Successfully!!!!!!!!!!!!!!!!!!!!!')
        print(is_test_valid)
        examples = _get_examples_from_experiments(experiments, dataset_folder)
        if frames_filter is not None:
            examples = _filter_examples(examples, frames_filter)
        return cls(examples, dataset_folder, data_hparams, is_test_valid=is_test_valid)

class IndoorFlying(MvsecDataset):
    @staticmethod
    def disparity_to_depth(disparity_image):
        unknown_disparity = disparity_image == float('inf')
        depth_image = \
            dataset_constants.FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (
            disparity_image + 1e-7)
        depth_image[unknown_disparity] = float('inf')
        return depth_image

    @staticmethod
    def split(dataset_folder, split_number, data_hparams,random_seed=0):
        """Creates training, validation and test sets.

        Args:
            dataset_folder: path to dataset.
            split_number: number of split (same as number of test sequence).
        """
        if split_number == 1:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [1]}, data_hparams,
                                           FRAMES_FILTER_FOR_TEST)
            
            # print('Successfully!!!!!!!!!!!!!!!!!!!!!')
            dataset.shuffle(random_seed)
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            # return (IndoorFlying.dataset(dataset_folder,
            #                              {'indoor_flying': [2, 3]}, data_hparams,
            #                              FRAMES_FILTER_FOR_TRAINING),
            #         validation_set, dataset)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [2, 3]}, data_hparams,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 2:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [2]}, data_hparams,
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle(random_seed)
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 3]}, data_hparams,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        elif split_number == 3:
            dataset = IndoorFlying.dataset(dataset_folder,
                                           {'indoor_flying': [3]}, data_hparams,
                                           FRAMES_FILTER_FOR_TEST)
            dataset.shuffle(random_seed)
            validation_set, test_set = dataset.split_into_two(
                first_subset_size=200)
            return (IndoorFlying.dataset(dataset_folder,
                                         {'indoor_flying': [1, 2]}, data_hparams,
                                         FRAMES_FILTER_FOR_TRAINING),
                    validation_set, test_set)
        else:
            raise ValueError('Test sequence should be equal to 1, 2 or 3.')


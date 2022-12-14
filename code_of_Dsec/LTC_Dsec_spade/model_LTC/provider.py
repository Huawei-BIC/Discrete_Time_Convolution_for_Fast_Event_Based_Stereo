from pathlib import Path
import random
import torch
import PIL.Image 
import cv2
import numpy as np
# from . import sequence
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,path, pre_frame=6):
        self.path = path
        self.pre_frame = pre_frame

    def __len__(self):
        return len(self.path)

    def get_example(self, index):
        
        left_path = self.path[index]
        right_path = left_path.parent.parent/'right'/left_path.parts[-1]
        
        gt_name = left_path.parts[-1][:6]+'.png'
        gt_path = Path('/media/HDD1/personal_files/zkx/datasets/Dsec/train/train_disparity/')/left_path.parent.parent.parts[-1]/'disparity'/'event'/gt_name

        frame_index = int(left_path.stem)
        left_eq, right_eq = [], [] 
        first_index = int(max(1,(frame_index-self.pre_frame*2)))# this should be 1, because we don`t have 000000.npy, and the first index of npy should be 000002.npy.
        for previous_frame_index in range(frame_index, first_index, -2):

            left_eq.append(torch.from_numpy(np.load(left_path.parent/'{:06d}.npy'.format(previous_frame_index)))) 
            right_eq.append(torch.from_numpy(np.load(right_path.parent/'{:06d}.npy'.format(previous_frame_index)))) 
        
        if len(left_eq) < self.pre_frame:
            total_need = self.pre_frame-len(left_eq)
            # print(total_need)
            for i in range(total_need):
                left_copy_last_frame = left_eq[-1].clone()
                right_copy_last_frame = right_eq[-1].clone()
                left_eq.append(left_copy_last_frame)
                right_eq.append(right_copy_last_frame)
        
        
        left_event_queue = torch.cat(left_eq,0).float()
        right_event_queue = torch.cat(right_eq,0).float()

        disp_16bit = cv2.imread(str(gt_path), cv2.IMREAD_ANYDEPTH)
        disp_16bit = disp_16bit.astype('float32')/256.0
        valid_disp = (disp_16bit > 0)
        disp_16bit[~valid_disp] = float('inf')
        gt_disparity = torch.from_numpy(np.array(disp_16bit))

        return {
                'left':{
                    'event_queue':left_event_queue,
                    'disparity_image': gt_disparity
                    },

                'right':{
                'event_queue': right_event_queue
                    },

                'frame_index': int(left_path.parts[-1][:6]),
                }


    def __getitem__(self, index):

        assert index<len(self.path)
        return self.get_example(index) 



class DatasetProvider:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def get_train_and_valid_dataset(self, pre_frame, all_data):
        train_path = self.dataset_path / 'train'
    
        assert self.dataset_path.is_dir(), str(self.dataset_path)
        assert train_path.is_dir(), str(train_path)

        partial_train_sequences = list()
        partial_valid_sequences = list()

        valid_seq_name = ['interlaken_00_c', 'interlaken_00_d', 'zurich_city_09_c', 'zurich_city_09_e', 'zurich_city_00_a', 'zurich_city_01_a','zurich_city_02_a', 'zurich_city_02_d','zurich_city_04_a', ]
        
        if all_data:
            for child in train_path.iterdir():
                one_side = child / 'left'
                if child.parts[-1] in valid_seq_name:
                    for pic_path in one_side.iterdir():
                        partial_valid_sequences.append(pic_path)

                for pic_path in one_side.iterdir():
                    partial_train_sequences.append(pic_path)

        else:
            for child in train_path.iterdir():
                one_side = child / 'left'
                if child.parts[-1] in valid_seq_name:
                    for pic_path in one_side.iterdir():
                        partial_valid_sequences.append(pic_path)

                else:
                    for pic_path in one_side.iterdir():
                        partial_train_sequences.append(pic_path)
        
        
        # for child in train_path.iterdir():
        #     one_side = child / 'left'
        #     if child.parts[-1] in valid_seq_name:
        #         for pic_path in one_side.iterdir():
        #             partial_valid_sequences.append(pic_path)

        #     for pic_path in one_side.iterdir():
        #         partial_train_sequences.append(pic_path)

        train_set = dataset(partial_train_sequences, pre_frame)
        print("train_set.len",len(train_set))
        valid_set = dataset(partial_valid_sequences, pre_frame)
        print("valid_set.len",len(valid_set))
        

        return train_set, valid_set



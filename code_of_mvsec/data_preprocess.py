import numpy as np
import os
from tqdm.notebook import tqdm

# The SBT method
def sbt_frame(events, start_time, height=260, width=346 , ms_per_channel=10, channel_per_frame=5):
        num_events = events.shape[0]
        final_frame = np.zeros((channel_per_frame,height,width))
        for i in range(num_events):
            channel_index =int((start_time - events[i,0])*1000 // ms_per_channel)
            x_position = int(events[i,1])
            y_position = int(events[i,2])
            polarity = int(events[i,3])
            assert 0<= channel_index <= 4 ,"channel_index should be [0,4]"
            final_frame[channel_index,y_position,x_position] += polarity
        return np.sign(final_frame)
    
def get_paths(experiment_number):
    paths = {}
    root = "indoor_flying_%d/" % experiment_number
    paths['data_folder0'] = root + "event0/"
    paths['data_folder1'] = root + "event1/"
    paths['experiment_folder0'] = root + "event0_10ms_frame/"
    paths['experiment_folder1'] = root + "event1_10ms_frame/"
    paths['timestamp'] = root + 'timestamps.txt'
    if not os.path.exists(paths['experiment_folder0']):
        os.makedirs(paths['experiment_folder0'])
    if not os.path.exists(paths['experiment_folder1']):
        os.makedirs(paths['experiment_folder1'])
    return paths

experiment_numbers = [1,2,3,4]

pbr0 = tqdm(total = len(experiment_numbers))
for i in experiment_numbers:
    print("processing indoor_flying%d" % i)
    paths = get_paths(i)
    timestamp_file = np.loadtxt(paths['timestamp'])
    num_image = len(timestamp_file)
    pbr = tqdm(total = num_image)
    for i in range(num_image):
        file_name = "%06d.npy" % i
        events0 = np.load(paths["data_folder0"] + file_name)
        events1 = np.load(paths["data_folder1"] + file_name)

        start_time = timestamp_file[i]
        event_frame0 = sbt_frame(events0,start_time)
        event_frame1 = sbt_frame(events1,start_time)

        np.save(paths["experiment_folder0"] + file_name,event_frame0)
        np.save(paths["experiment_folder1"] + file_name,event_frame1)
        pbr.update(1)
    
    pbr.close()
    pbr0.update(1)


pbr0.close()
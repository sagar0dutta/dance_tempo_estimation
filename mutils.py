
import os
import pickle
import numpy as np
import pandas as pd
from foot_module import onset_calculations
from IPython.display import display
from scipy.signal import find_peaks

OnsetProcessor = onset_calculations.OnsetProcessor()

class DataHandler:
    def __init__(self):
        self.drum_onsets = None
        self.start_t = None
        self.end_t = None

    def load_and_process_data(self, file_name, mode = 'gr', drum = 'J2', section_idx=0):
        

        file_path = f"./mocap_mvnx/{file_name}"
        file_name = os.path.basename(file_path).split(".")[0]

        pickle_path = f'./output/motion_data_pkl/{file_name}.pkl'
            
        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as file:
                motion_data = pickle.load(file)
            print(f"Loaded {file_name}.pkl")
            
        onset_filename = file_name.replace("_T", "")
        danceanno_filename = file_name.replace("_T", "_Dancers")
        mcycle_filename = file_name.replace("_T", "_C")

        drum_onsets_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/drum_onsets/{onset_filename}.csv"
        dance_anno_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/Dataset_V2/{danceanno_filename}.csv"
        mcycle_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/metric_cycles/{mcycle_filename}.csv"

        # Load meter cycle onsets
        df_metric = pd.read_csv(mcycle_path)
        loaded_mcycle_onsets = np.round(df_metric["Selected Onset"].to_numpy(), 3)
        
        # Load drum onsets
        drum_df = pd.read_csv(drum_onsets_path)
        column_b = drum_df[drum].dropna()
        self.drum_onsets = column_b.to_numpy()

        # Load dance annotations
        anno_df = pd.read_csv(dance_anno_path)
        category_df = anno_df.groupby('mocap')
        try:
            category_df = category_df.get_group(mode)
        except KeyError:
            raise ValueError(f"Group '{mode}' not found in the dataset.")
    
        # category_df = category_df.get_group(mode)
        category_df = category_df.reset_index(drop=True)
        # display(category_df)

        start_f = np.round(category_df.iloc[section_idx, 6]*240).astype(int)
        end_f = np.round(category_df.iloc[section_idx, 7]*240).astype(int)
        self.start_t = start_f/240 
        self.end_t = end_f/240
        
        section_data = OnsetProcessor.onset_calculations(category_df, loaded_mcycle_onsets)

        section = section_data[section_idx]
        print("Total Sections:", len(section))
        # print("index 0 for first section")
        
        section_name, section_info = list(section.items())[0]
        section_onset_data = section_info["section_onset_data"]
        cycle_onsets = section_onset_data["cycle_onsets"]
        
        # Calculate beat onsets from cycle onsets
        beat_ref = self.calculate_beat_onsets(cycle_onsets)
       
        bpm = self.calc_tempo_from_onsets(beat_ref[(beat_ref >= self.start_t) & (beat_ref <= self.end_t)])
        # print(f"{drum} tempo for the current section: {bpm:.2f} BPM")
        
        return motion_data, self.drum_onsets, start_f, end_f, self.start_t, self.end_t, cycle_onsets, beat_ref, bpm
    
    @staticmethod
    def calculate_beat_onsets(cycle_onsets):
        beat_ref = []
        for i in range(len(cycle_onsets) - 1):
            start = cycle_onsets[i]
            end = cycle_onsets[i + 1]
            # Generate beat onsets by dividing the interval into 4 equal parts
            beat_onsets = np.linspace(start, end, num=5)[:-1]
            beat_ref.extend(beat_onsets)
        return np.round(np.array(beat_ref), 3)
    @staticmethod
    def calc_tempo_from_onsets(onset):
        iois = np.diff(onset)
        mean_ioi = np.mean(iois)
        bpm = 60 / mean_ioi
        return bpm
    
    def load_and_process_feetdata(self, file_name, mode = 'gr', drum = 'J2', section_idx=0):
        

        file_path = f"./mocap_mvnx/{file_name}"
        file_name = os.path.basename(file_path).split(".")[0]

        pickle_path = f'./output/motion_data_pkl/{file_name}.pkl'
            
        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as file:
                motion_data = pickle.load(file)
            print(f"Loaded {file_name}.pkl")

        onset_filename = file_name.replace("_T", "")
        danceanno_filename = file_name.replace("_T", "_Dancers")
        mcycle_filename = file_name.replace("_T", "_C")

        drum_onsets_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/drum_onsets/{onset_filename}.csv"
        dance_anno_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/Dataset_V2/{danceanno_filename}.csv"
        mcycle_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/metric_cycles/{mcycle_filename}.csv"

        # Load meter cycle onsets
        df_metric = pd.read_csv(mcycle_path)
        loaded_mcycle_onsets = np.round(df_metric["Selected Onset"].to_numpy(), 3)
        
        # Load drum onsets
        drum_df = pd.read_csv(drum_onsets_path)
        column_b = drum_df[drum].dropna()
        self.drum_onsets = column_b.to_numpy()

        # Load dance annotations
        anno_df = pd.read_csv(dance_anno_path)
        category_df = anno_df.groupby('mocap')
        try:
            category_df = category_df.get_group(mode)
        except KeyError:
            raise ValueError(f"Group '{mode}' not found in the dataset.")
    
        category_df = category_df.reset_index(drop=True)
        # display(category_df)

        # load feet onsets
        feet_length = len(motion_data['position']['SEGMENT_RIGHT_FOOT']) # size (n, 3)
        feet_onsets_path = f"/itf-fi-ml/home/sagardu/djembe_drive/sgr_pyspace/logs/{file_name}/onset_info/{file_name}_both_feet_onsets.csv"
        feet_df = pd.read_csv(feet_onsets_path, usecols=[1])
        feet_frames = feet_df.values.flatten()

        feet_onsets = np.zeros(feet_length)
        feet_onsets[feet_frames] = 1
        feet_onsets = np.reshape(feet_onsets, (-1, 1))
        
        #########
        start_f = np.round(category_df.iloc[section_idx, 6]*240).astype(int)
        end_f = np.round(category_df.iloc[section_idx, 7]*240).astype(int)
        self.start_t = start_f/240 
        self.end_t = end_f/240
        
        section_data = OnsetProcessor.onset_calculations(category_df, loaded_mcycle_onsets)

        section = section_data[section_idx]
        print("Total Sections:", len(section))
        print("index 0 for first section")
        
        section_name, section_info = list(section.items())[0]
        section_onset_data = section_info["section_onset_data"]
        cycle_onsets = section_onset_data["cycle_onsets"]
        
        # Calculate beat onsets from cycle onsets
        beat_ref = self.calculate_beat_onsets(cycle_onsets)
       
        bpm = self.calc_tempo_from_onsets(beat_ref[(beat_ref >= self.start_t) & (beat_ref <= self.end_t)])
        print(f"{drum} tempo for the current section: {bpm:.2f} BPM")
        
        return motion_data, self.drum_onsets, start_f, end_f, self.start_t, self.end_t, cycle_onsets, beat_ref, bpm, feet_onsets
    
    
    def onsets_for_plotting(self, sensor_dir_change_onsets, estimated_beat_pulse, novelty_length):

        drum_ref = self.drum_onsets[(self.drum_onsets >= self.start_t) & (self.drum_onsets <= self.end_t)]
        half_beats = (drum_ref[:-1] + drum_ref[1:]) / 2
        drum_subref = np.sort(np.concatenate((drum_ref, half_beats)))

        # raw onsets
        onsets_frm = np.where(sensor_dir_change_onsets > 0)[0]     # Frames idx of onsets
        onsets_t = onsets_frm/240
        dance_onset = onsets_t[(onsets_t >= self.start_t) & (onsets_t <= self.end_t)]
        dance_onset_iois = np.diff(dance_onset)
        
        # estimated beats
        time = np.arange(novelty_length) / 240
        peaks, _ = find_peaks(estimated_beat_pulse)  # , prominence=0.02
        peaks_t = time[peaks]
        estimated_beats = peaks_t[(peaks_t >= self.start_t) & (peaks_t <= self.end_t)]
        estimated_beats_iois = np.diff(estimated_beats)
        
        return dance_onset, estimated_beats, drum_ref, dance_onset_iois, estimated_beats_iois
    

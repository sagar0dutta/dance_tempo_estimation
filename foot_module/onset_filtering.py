
import numpy as np


class Filtering:
    def __init__(self, mocap_fps = 240):
        self.mocap_fps = mocap_fps


    def filter_minima(self, maxima_foot, minima_foot, Rshifted_ankle_pos_data,
                    pos_diff_threshold=0.02, max_threshold=0.02, time_diff_threshold=0.16, mocap_fps = 240):
        
        # filters minima based on position and time thresholds
        
        # Convert time difference threshold from seconds to frames
        frame_threshold = int(time_diff_threshold * self.mocap_fps)  # Convert time threshold to frames
        
        valid_minima = []
        
        for min_frame in minima_foot:
            # Find the nearest maximum before the minimum
            maxima_before_min = [max_frame for max_frame in maxima_foot if max_frame < min_frame]
            if not maxima_before_min:
                continue  # No maxima before this minimum
            
            # Consider the nearest maximum
            nearest_max_frame = maxima_before_min[-1]
            
            # Check the positional difference condition
            pos_diff = abs(Rshifted_ankle_pos_data[nearest_max_frame] - Rshifted_ankle_pos_data[min_frame])
            if pos_diff < pos_diff_threshold:
                continue
            
            # Check the threshold for maxima
            if Rshifted_ankle_pos_data[nearest_max_frame] <= max_threshold:
                continue
            
            # Check the temporal difference condition
            time_diff = abs(nearest_max_frame - min_frame)
            if time_diff < frame_threshold:
                continue

            valid_minima.append(min_frame)
        
        return np.array(valid_minima)


    def create_binary_foot_contact(self, pos_data, foot_events, b_value = 1, pos_thres = 0.03):
        # Version 2
        contact_binary = np.zeros(len(pos_data))
        foot_data = np.empty((0, 2))

        for foot_event in foot_events:
            current_lift, ground_contact, maxima_frames, maxima_value, _, _ = foot_event         # , minima_frames, minima_value
            
            if len(maxima_frames) > 1:
                if maxima_value[-1] >= pos_thres:
                    contact_binary[current_lift: ground_contact] = b_value
                    foot_row = np.array([current_lift, ground_contact])
                    foot_data = np.vstack([foot_data, foot_row])
                    # print(maxima_frames, maxima_value, ground_contact)
            
            else:
                if maxima_value >= pos_thres:
                    contact_binary[current_lift: ground_contact] = b_value
                    foot_row = np.array([current_lift, ground_contact])
                    foot_data = np.vstack([foot_data, foot_row])
                    # print(maxima_frames, maxima_value, ground_contact)
                    
        return contact_binary, foot_data


    def select_first_foot_contact(self, ankle_onsets, toe_onsets, ank_toe_thres_sec = 0.20):

        # identifies the first contact (ankle or toe) within a specified threshold
        
        threshold_frames = int(ank_toe_thres_sec * self.mocap_fps)

        # Initialize lists to store results
        selected_onsets = []
        onset_type = []

        # Keep track of which heel and toe onsets are already used
        used_heel_onsets = set()
        used_toe_onsets = set()

        toe_heel_diff = []
        heel_toe_diff = []

        # Case 1: Simultaneous or near-simultaneous heel and toe contact
        for toe in toe_onsets:
            # Check for heel contact within 38 frames and ensure it's not used already
            heel_in_window = [heel for heel in ankle_onsets if heel > toe and heel <= toe + threshold_frames and heel not in used_heel_onsets]
            if len(heel_in_window) > 0:
                toe_heel_differences = [heel-toe for heel in heel_in_window]
                toe_heel_diff.extend(toe_heel_differences)
                
                selected_onsets.append(toe)
                onset_type.append('toe -> heel (simultaneous)')
                used_toe_onsets.add(toe)
                used_heel_onsets.add(heel_in_window[0])  # Mark the heel onset as used

        for heel in ankle_onsets:
            # Check for toe contact within 38 frames and ensure it's not used already
            toe_in_window = [toe for toe in toe_onsets if toe > heel and toe <= heel + threshold_frames and toe not in used_toe_onsets]
            if len(toe_in_window) > 0:
                heel_toe_differences = [toe-heel for toe in heel_in_window]
                heel_toe_diff.extend(heel_toe_differences)
                
                selected_onsets.append(heel)
                onset_type.append('heel -> toe (simultaneous)')
                used_heel_onsets.add(heel)
                used_toe_onsets.add(toe_in_window[0])  # Mark the toe onset as used

        # Case 2: Heel contact only (no toe contact within threshold_frames)
        for heel in ankle_onsets:
            if heel not in used_heel_onsets:
                toe_in_window = [toe for toe in toe_onsets if toe > heel and toe <= heel + threshold_frames]
                if len(toe_in_window) == 0:  # No toe contact within the window
                    selected_onsets.append(heel)
                    onset_type.append('heel (isolated)')
                    used_heel_onsets.add(heel)

        # Case 3: Toe contact only (no heel contact within threshold_frames)
        for toe in toe_onsets:
            if toe not in used_toe_onsets:
                heel_in_window = [heel for heel in ankle_onsets if heel > toe and heel <= toe + threshold_frames]
                if len(heel_in_window) == 0:  # No heel contact within the window
                    selected_onsets.append(toe)
                    onset_type.append('toe (isolated)')
                    used_toe_onsets.add(toe)

        # Convert lists to numpy arrays for sorting
        selected_onsets = np.array(selected_onsets)
        onset_type = np.array(onset_type)

        # Sort based on selected_onsets (frame numbers)
        sorted_indices = np.argsort(selected_onsets)
        selected_onsets = selected_onsets[sorted_indices]
        onset_type = onset_type[sorted_indices]

        selected_onsets_t = selected_onsets/240

        # Output selected onsets and their corresponding type (sorted by frame number)
        # for onset, o_type in zip(selected_onsets, onset_type):
        #     print(f"Onset at frame {onset}: {o_type}")

        return selected_onsets, np.array(toe_heel_diff), np.array(heel_toe_diff)

    # def filter_onsets_by_threshold(self, R_foot_minima, threshold_s=0.16):
    #     # Removes the second onset if it is within the window  
        

    #     threshold_frames = int(threshold_s * self.mocap_fps)
    #     filtered_onsets = []
    #     skip_flag = False
    #     for i in range(0, len(R_foot_minima) - 1):
            
    #         if skip_flag == True:
    #             skip_flag = False
    #             continue
            
    #         if R_foot_minima[i+1] - R_foot_minima[i] <= threshold_frames:
    #             filtered_onsets.append(R_foot_minima[i])
    #             skip_flag = True    # skip for loop for i+1
                
    #         else:
    #             filtered_onsets.append(R_foot_minima[i])
                
    #     # To ensure the last onset is checked
    #     if not skip_flag:
    #         filtered_onsets.append(R_foot_minima[-1])
            
    #     return np.array(filtered_onsets)
    
    
    def filter_onsets_by_threshold(self, frame_onsets, threshold_s=0.10):
    # Removes any onsets that fall within the threshold window after the current onset
    
        window_frames = int(threshold_s * self.mocap_fps)
        filtered_onsets = []
        i = 0
        
        while i < len(frame_onsets):
            current_frame_onset = frame_onsets[i]
            end_frame = current_frame_onset + window_frames
            
            # Add the current onset to the filtered list
            filtered_onsets.append(current_frame_onset)
            
            # Skip all subsequent onsets that fall within the window
            j = i + 1
            while j < len(frame_onsets) and frame_onsets[j] <= end_frame:
                j += 1
            
            # Update the index to the next onset that is outside the window
            i = j
        
        return np.array(filtered_onsets)
    


    def filter_minima_between_footstep(self, rfoot_onsets, Rankle_filtered_minima, threshold_s=0.16):

        # Convert threshold from seconds to frames
        threshold_frames = int(threshold_s * self.mocap_fps)
        filtered_onsets = [] 
        
        # skip_flag = False
        for i in range(0, len(rfoot_onsets) - 1):
            
            filtered_onsets.append(rfoot_onsets[i])
            filtered_onsets.append(rfoot_onsets[i+1])
            temp_minima=[]
            
            for minima in Rankle_filtered_minima:
                if minima >= rfoot_onsets[i] and minima <= rfoot_onsets[i+1]:
                    temp_minima.append(minima)
            
            if len(temp_minima) == 1:
                # Check if this minima is too close to the first or second onset
                if abs(temp_minima[0] - rfoot_onsets[i]) <= threshold_frames:
                    del temp_minima[0]  # Remove if it's too close to the first onset
                elif abs(temp_minima[0] - rfoot_onsets[i+1]) <= threshold_frames:
                    del temp_minima[0]  # Remove if it's too close to the second onset

            if len(temp_minima) > 1:
                # Remove the first minima if it's too close to the first onset
                if abs(temp_minima[0] - rfoot_onsets[i]) <= threshold_frames:
                    del temp_minima[0]

                # Remove the last minima if it's too close to the second onset
                if abs(temp_minima[-1] - rfoot_onsets[i+1]) <= threshold_frames:
                    del temp_minima[-1]
            
            filtered_onsets.extend(temp_minima)
        
        filtered_onsets = np.sort(filtered_onsets)  
            
        return np.array(filtered_onsets)

import numpy as np
from scipy.signal import savgol_filter, argrelmin, argrelmax


class FootEventDetector:
    def __init__(self):
        pass

    def foot_detection(self, motiondata_section, foot_segment_name, foot_ground_pos, foot_ground_threshold = 0.005, time_threshold_seconds= 0.16, zheight_threshold = 0.00):

        foot_xyzpos_data = motiondata_section["position"][foot_segment_name]
        
        # resultant = np.linalg.norm(foot_xyzpos_data, axis=1)
        # foot_zpos_data = savgol_filter(resultant, 60, 0)
        
        foot_zpos_data = savgol_filter(foot_xyzpos_data[:,2], 60, 0)

        foot_vel_data = motiondata_section["velocity"][foot_segment_name]
        foot_acc_data = motiondata_section["acceleration"][foot_segment_name]


        shifted_foot_pos_data = self.apply_ground_threshold(foot_zpos_data, foot_ground_pos, foot_ground_threshold = foot_ground_threshold)      # ground position correction
        foot_xyzpos_corrected = np.hstack((foot_xyzpos_data[:,0:2], shifted_foot_pos_data[:, None]))     # xy + ground corrected z position

        foot_lift_onsets, foot_ground_contact_onsets, foot_maxima_onsets, foot_minima_onsets,  = self.detect_foot_endpoints(shifted_foot_pos_data, ground_contact_threshold = 0.001)     # Change threshold only if needed

        # Detect foot events
        foot_events = self.extract_foot_event_onsets(shifted_foot_pos_data, foot_lift_onsets, foot_maxima_onsets, foot_ground_contact_onsets, foot_minima_onsets)

        # Time threshold should be relative average beat period, in this case half of beat period
        foot_events_filtered, foot_lift_onsets_filtered, foot_ground_onsets_filtered, foot_maxima_filtered,foot_minima_filtered = self.filter_foot_events(shifted_foot_pos_data, foot_events, time_threshold_seconds= time_threshold_seconds, zheight_threshold = zheight_threshold)

        foot_event_data = []

        for i in range(len(foot_events_filtered)):

            current_lift, ground_contact, _, _, _, _ = foot_events_filtered[i]

            foot_position = foot_xyzpos_corrected[current_lift: ground_contact+1]
            foot_velocity = foot_vel_data[current_lift: ground_contact+1]        # velocity between foot lift and ground contact
            foot_acceleration = foot_acc_data[current_lift: ground_contact+1]    # acceleration between foot lift and ground contact
            
            foot_event_data.append({"foot_event_onsets": foot_events_filtered[i],
                                    "foot_position": foot_position,             # xy + ground corrected z position
                                    "foot_velocity": foot_velocity,             # array (n,3)
                                    "foot_acceleration": foot_acceleration}     # array (n,3)
                                )
        
        return {
        "foot_events_unfiltered": foot_events,   
        "foot_events_filtered": foot_events_filtered,
        "foot_lift_onsets_filtered": foot_lift_onsets_filtered,
        "foot_ground_onsets_filtered": foot_ground_onsets_filtered,
        "foot_maxima_filtered": foot_maxima_filtered,
        "foot_minima_filtered": foot_minima_filtered,
        "foot_event_filtered_data": foot_event_data,
        "shifted_foot_pos_data": shifted_foot_pos_data,
        "foot_vel_data": foot_vel_data
        }



    def apply_ground_threshold(self, pos_data, foot_ground_pos, foot_ground_threshold = 0.005):
        """
        Apply a ground threshold to position data, setting all values below the threshold equal to it and 
        shifting the data so the ground position becomes zero.

        Parameters:
        pos_data (np.array): The array of position data.
        foot_ground_pos (float): The foot's ground position value in meters.
        foot_ground_threshold (float): The threshold below which all values will be adjusted.

        Returns:
        np.array: The shifted position data with the threshold applied.
        """
        # Calculate the threshold
        threshold = foot_ground_pos + foot_ground_threshold
        
        adjusted_pos = np.copy(pos_data)
        adjusted_pos[adjusted_pos <= threshold] = threshold
        
        # Shift the data so the foot ground position is zero
        shifted_pos = adjusted_pos - threshold

        return shifted_pos


    def detect_foot_endpoints(self, shifted_pos, ground_contact_threshold = 0.001):
        ''' Detects endpoints of a movement trajectory'''
        
        
        minima = argrelmin(shifted_pos, order=1)[0]     # Adjust order to detect closer valleys or minima             # Find minima of foot contact
        maxima = argrelmax(shifted_pos, order=1)[0]     # Adjust order to detect closer peaks or maxima               # Find maxima of foot contact
        foot_minima_onsets = np.sort(minima)                            # Sort the frames idx ascending order
        foot_maxima_onsets = np.sort(maxima)                            # Sort the frames idx ascending order
        # foot_maxima_value = shifted_pos[foot_maxima_onsets]
                                                    
        above_threshold = np.where(shifted_pos > ground_contact_threshold, True, False)                # ground threshold for detecting ground contact (in meters)
        ground_contact_onsets = np.where(np.diff(above_threshold.astype(int)) == -1)[0]                # first ground contact onsets after lifting foot
        foot_lift_onsets = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
        
        
        # The statement checks for every value in the shifted_pos array whether it's greater than 
        # ground_contact_threshold. If a value is greater, the corresponding position in above_threshold will be True. 
        # If not, it will be False.

        # Now, if above_threshold is form [0, 1, 1, 0], then np.diff() would result in [1, 0, -1].
        # 1 indicates a transition from 0 (below threshold) to 1 (above threshold).
        # -1 indicates a transition from 1 (above threshold) to 0 (below threshold).
        
        return foot_lift_onsets, ground_contact_onsets, foot_maxima_onsets, foot_minima_onsets

    def extract_foot_event_onsets(self, shifted_foot_pos_data, foot_lift_onsets, foot_maxima_onsets, foot_ground_contact_onsets, foot_minima_onsets):

        foot_events = []  # To store the result

        # Iterate over foot_lift_onsets
        for i in range(len(foot_lift_onsets) - 1):
            current_lift = foot_lift_onsets[i]
            next_lift = foot_lift_onsets[i + 1]

            # Find foot ground contact onsets between foot_lift_onsets[i] and foot_lift_onsets[i+1]
            for ground_contact in foot_ground_contact_onsets:
                if current_lift < ground_contact < next_lift:
                    
                    # Find foot maxima onsets between foot_lift_onsets[i] and the found ground contact onset
                    maxima_in_range = [maxima for maxima in foot_maxima_onsets if current_lift < maxima < ground_contact]
                    
                    if len(maxima_in_range) > 0:
                        maxima_pos_value = shifted_foot_pos_data[maxima_in_range]
                    else:
                        maxima_pos_value = None
                    

                    # Check if there are more than one maxima, then get the minima
                    if len(maxima_in_range) > 1:
                        minima_in_range = [minima for minima in foot_minima_onsets if current_lift < minima < ground_contact]
                        minima_pos_value = shifted_foot_pos_data[minima_in_range]
                    
                    else:
                        minima_in_range = np.array([])
                        minima_pos_value = np.array([])
                    
                    
                    
                    foot_events.append((current_lift,               # foot lift onset
                                        ground_contact,             # ground contact onset
                                        np.array(maxima_in_range),  # maxima onset
                                        maxima_pos_value,           # maxima position value
                                        np.array(minima_in_range),  # minima onset
                                        minima_pos_value)           # minima position value
                                    )
        
        # print("Total Foot events:", len(foot_events))

        return foot_events
        

    def filter_foot_events(self, pos_data, foot_events, time_threshold_seconds= 0.16, zheight_threshold = 0.03):

        ##### FILTERING BASED ON Time threshold (sec) between foot lift and foot ground contact 
        # AND z-position height threshold (meters)

        time_threshold_frames = round(time_threshold_seconds * 240)         # fps = 240

        foot_events_filtered = []
        foot_lift_onsets_filtered = []
        foot_ground_contact_onsets_filtered = []
        foot_maxima_onsets = []
        foot_minima_onsets = []
        
        for foot_event in foot_events:
            current_lift, ground_contact, maxima_array, maxima_value, minima_array, minima_value = foot_event      # Onsets frame
            
            # Filter based on time (ensure the time difference is greater than the threshold)
            if (ground_contact - current_lift) >= time_threshold_frames:
                
                # Filter maxima based on position value from pos_data
                maxima_array_filtered = [maxima for maxima in maxima_array if pos_data[maxima] >= zheight_threshold]
                
                # code here to filter minima onsets
                    # use slope threshold between maxima value and minima value
                
                # Append only if there are valid maxima after filtering
                if len(maxima_array_filtered) > 0:
                    foot_events_filtered.append((current_lift, ground_contact, maxima_array, maxima_value,  minima_array, minima_value))
                    
                    foot_lift_onsets_filtered.append(current_lift)
                    foot_ground_contact_onsets_filtered.append(ground_contact)
                    foot_maxima_onsets.extend(maxima_array_filtered)
                    foot_minima_onsets.extend(minima_array)
                    
        # print("Total filtered foot events:", len(foot_events_filtered))
        
        return foot_events_filtered, np.array(foot_lift_onsets_filtered), np.array(foot_ground_contact_onsets_filtered), np.array(foot_maxima_onsets),np.array(foot_minima_onsets)
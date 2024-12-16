
import numpy as np


class OnsetProcessor:
    def __init__(self):
        pass
        # self.category_df = category_df
        # self.loaded_mcycle_onsets = loaded_mcycle_onsets
        
    def onset_calculations(self, category_df, loaded_mcycle_onsets,choose_nb_onset_to_make_block = 2):

        total_sections = category_df.shape[0]
        # print("total sections:", total_sections)
        
        section_data = []       # Each piece is divided into sections
        for section_idx in range(total_sections):

            row = section_idx
            # For each section
            timecode1 = category_df.iloc[row, 0]         # Column "Start"
            timecode2 = category_df.iloc[row, 1]         # Column "End"
            category = category_df.iloc[row, 5]          # Column mocap
            start_sec = category_df.iloc[row, 6]         # Column "Start (in sec)"
            end_sec = category_df.iloc[row, 7]           # Column "End (in sec)"
            duration_sec = category_df.iloc[row, 8]      # Column Length (sec)

            cycle_onsets = np.array([value for value in loaded_mcycle_onsets if start_sec <= value <= end_sec])
            cycle_period_list = np.diff(cycle_onsets)                   # list of cycle periods for all blocks
            

            all_window_onsets_originaltime = self.create_blocks(cycle_onsets, choose_nb_onset_to_make_block)      # Create blocks without overlap

            # all_window_onsets = create_blocks_overlap(cycle_onsets, choose_nb_onset_to_make_block)        # 
            window_period_list = [round(v2-v1,3) for v1,v2 in all_window_onsets_originaltime]  
            total_blocks = len(all_window_onsets_originaltime)

            # print(f"total cycles in section {section_idx+1}:", total_blocks)
            
            
            section_meta_data = {   # For each section of the performance piece
                "start_timestamp": timecode1,   # section start
                "end_timestamp": timecode2,     # section end
                "category": category,
                "start": start_sec,
                "end": end_sec,
                "duration": duration_sec,
            }

            section_onset_data = {      # For each section of the performance piece
                "cycle_onsets": cycle_onsets,
                "cycle_period_list": cycle_period_list,
                "all_window_onsets": all_window_onsets_originaltime,
                "window_period_list": window_period_list,
                "total_blocks": total_blocks,
            }

            section_temp = {f"Section_{section_idx+1}":{"section_meta_data": section_meta_data,
                            "section_onset_data": section_onset_data}}
            
            section_data.append(section_temp)
        
        
        return section_data

    def create_blocks(self, onsets, nb_block):    # No cycle overlap
        ''' Creates (start, end) onset for each cycle'''
        all_blocks = []
        step_size = nb_block - 1  
        
        # Loop through onsets
        for i in range(0, len(onsets), step_size):
            start = onsets[i]  # First onset of the block or cycle
            end = onsets[min(i + nb_block - 1, len(onsets) - 1)] 
            
            # Append the tuple (start, end)
            all_blocks.append((round(start, 3), round(end, 3)))
            
            # Stop at the end of the list
            if i + nb_block >= len(onsets):
                break

        return all_blocks

    def create_blocks_overlap(self, onsets, choose_nb_onset_to_make_block):
        # Create pairs by shifting the onsets by choose_nb_onset_to_make_block
        return [(onsets[i], onsets[i + choose_nb_onset_to_make_block - 1]) for i in range(len(onsets) - choose_nb_onset_to_make_block + 1)]





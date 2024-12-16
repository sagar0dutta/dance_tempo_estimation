import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pydub import AudioSegment
from pydub.generators import Triangle


class Utils_func:
    def __init__(self, section_data, file_name, foot_segment_name, mocap_mode, mode_folder_path, kde_folder_path):
        self.section_data = section_data
        self.total_section = len(section_data)
        self.file_name = file_name
        self.foot_segment_name = foot_segment_name
        self.mocap_mode = mocap_mode
        self.mode_folder_path = mode_folder_path
        self.kde_folder_path = kde_folder_path

    def collect_foot_onsets_per_cycle(self, b_foot_cleaned_t):

        feet_onsets_per_section = []
        feet_onsets_all = []
        section_data_list = []
        total_feet_onsets = 0
        for section_idx in range(self.total_section):
            
            section = self.section_data[section_idx]  # Change the index if you want another section
            
            # Extract the section name and its details
            section_name, section_details = next(iter(section.items()))  # Unpacking the first item
            section_meta_data = section_details["section_meta_data"]
            section_onset_data = section_details["section_onset_data"]
            
            # Access section onset data
            section_onset_data = section_details["section_onset_data"]
            cycle_period_list = section_onset_data["cycle_period_list"]
            all_window_onsets = section_onset_data["all_window_onsets"]

            
            peaks_per_cycle_all = []
            for i in range(len(all_window_onsets)):
                a = all_window_onsets[i][0]
                b = all_window_onsets[i][1]
                
                peaks_per_cycle = np.array([onset -a for onset in b_foot_cleaned_t if a <= onset <= b])      # foot onsets for a cycle
                peaks_per_cycle_all.append(peaks_per_cycle/cycle_period_list[i])                            # list of foot onsets for all cycles in a section  
                
                total_feet_onsets += len(peaks_per_cycle)   # for stat
                feet_onsets_all.extend([onset for onset in b_foot_cleaned_t if a <= onset <= b])
            
            feet_onsets_per_section.append(peaks_per_cycle_all)
            
            # Save the sections info for a mode
            section_info = {
                    "Section_Name": section_name,
                    "Start_Timestamp": section_meta_data["start_timestamp"],
                    "End_Timestamp": section_meta_data["end_timestamp"],
                    "Mode": section_meta_data["category"],
                    "Start_sec": section_meta_data["start"],
                    "End_sec": section_meta_data["end"],
                    "Duration": section_meta_data["duration"],
                    "Cycle_Onsets": section_onset_data["cycle_onsets"],
                    "Cycle_Periods": section_onset_data["cycle_period_list"],
                    "Total_Cycles": section_onset_data["total_blocks"],
                    "total_feet_onsets": total_feet_onsets
                }
            section_data_list.append(section_info)
            
  
        # Save the feet onsets cycles per sections in the mode folder
        with open(os.path.join(self.mode_folder_path, f"{self.file_name}_{self.foot_segment_name}_Onsets_section_cycles.pkl") , 'wb') as f:
            pickle.dump(feet_onsets_per_section, f)

        # save section data stats
        df_section_data_list = pd.DataFrame(section_data_list)
        df_feet_onsets_all = pd.DataFrame(feet_onsets_all)
        
        df_section_data_list.to_csv(os.path.join(self.mode_folder_path, f"{self.file_name}_{self.foot_segment_name}_{self.mocap_mode}_sections_details.csv"), index=False)
        df_feet_onsets_all.to_csv(os.path.join(self.mode_folder_path, f"{self.file_name}_{self.foot_segment_name}_{self.mocap_mode}_feet_onsets_all.csv"), index=False)

        return feet_onsets_per_section


    def kde_plot_save(self, feet_onsets):

        time_range = np.linspace(0, 1, 1000)
        total_density = np.zeros_like(time_range)
        valid_kde_count = 0
        print(f"Saving Average KDE : {self.file_name}, {self.foot_segment_name}, Mode: {self.mocap_mode}")
        for i in range(self.total_section):
            section = self.section_data[i]
            _, section_details = next(iter(section.items()))
            section_onset_data = section_details["section_onset_data"]
            total_blocks = section_onset_data["total_blocks"]
            
            for j in range(total_blocks):

                peak_locations_time = np.array(feet_onsets[i][j])
                
                if len(peak_locations_time) < 2:
                    continue

                try:
                    kde = gaussian_kde(peak_locations_time, bw_method= 0.04)
                    density = kde(time_range)
                
                except np.linalg.LinAlgError:
                    print(f"LinAlgError encountered for block {j}. Skipping KDE for this block.")
                    continue
                
                # Accumulate the density for averaging later
                total_density += density
                valid_kde_count += 1

        # average_density = total_density / valid_kde_count
        
        if valid_kde_count > 0:
            average_density = total_density / valid_kde_count
        else:
            average_density = np.zeros_like(total_density)
        
        # save kde
        
        df_avg_kde = pd.DataFrame(average_density)
        df_avg_kde.to_csv(os.path.join(self.kde_folder_path, f"{self.file_name}_{self.foot_segment_name}_Mode_{self.mocap_mode}_avgkde_onsets.csv"), index=False)
        
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(time_range, average_density)
        ax.set_title(f"Average KDE\n{self.file_name} \n {self.foot_segment_name}, Mode: {self.mocap_mode}")
        ax.set_xlabel("Normalized cycle")
        ax.set_ylabel('Density')

        # Add beat positions as vertical lines
        for onst in [0, 0.25, 0.5, 0.75]:  # Beat positions
            ax.axvline(x=onst, color='g', linestyle='--', ymin=0.01, ymax=2)

        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        
        fig.savefig(os.path.join(self.kde_folder_path, f"{self.file_name}_{self.foot_segment_name}_Mode_{self.mocap_mode}.png") , dpi = 200)
        plt.close(fig)
        print("Saving completed")
        return average_density


    def kde_subplot_save(self, kde_list, modes, kde_folder_path):
        # kde_list has three kde array for gr in and au
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)  # Create 3 subplots
        time_range = np.linspace(0, 1, 1000)

        print("Saving kde subplot...")
        for i, (average_density, mode) in enumerate(zip(kde_list, modes)):  # Iterate over the densities and modes
            axes[i].plot(time_range, average_density)
            axes[i].set_title(f"Average KDE\n{self.file_name} \n {self.foot_segment_name}, Mode: {mode}")
            axes[i].set_xlabel("Normalized cycle")
            axes[i].set_ylabel('Density')

            # Add beat positions as vertical lines
            for onst in [0, 0.25, 0.5, 0.75]:
                axes[i].axvline(x=onst, color='g', linestyle='--', ymin=0.01, ymax=2)

            axes[i].minorticks_on()
            axes[i].grid(which='minor', alpha=0.2)
            axes[i].grid(which='major', alpha=0.5)

        # Save the figure
        
        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.savefig(os.path.join(kde_folder_path, f"{self.file_name}_{self.foot_segment_name}_Modes_Subplot.png"), dpi=200)
        plt.close()
        print("Saving completed")

    # Function to calculate average KDE and save subplots
    def kde_per_piece_dancer_subplot_save(self, kde_dict, entity_name, modes, save_folder):
        """
        kde_dict: Dictionary containing KDE arrays for each mode (gr, in, au).
        entity_name: The name of the piece or dancer.
        modes: List of modes (gr, in, au).
        save_folder: Folder path where plots should be saved.
        entity_type: String to define whether it's a 'piece' or 'dancer' (for the plot title).
        """

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)  # Create 3 subplots
        time_range = np.linspace(0, 1, 1000)  # Assuming KDE is normalized over this range
        print(f"Saving Average KDE per Piece/Dance : {entity_name}, {self.foot_segment_name}")
        # Calculate average KDE for each mode
        for i, mode in enumerate(modes):
            kde_arrays = kde_dict[mode]
            
            # Check if there is data in the list, otherwise skip plotting
            if kde_arrays:
                # Compute the average of the KDE arrays for the current mode
                average_kde = np.mean(np.array(kde_arrays), axis=0)
                
                df_avg_kde_dancer = pd.DataFrame(average_kde)
                df_avg_kde_dancer.to_csv(os.path.join(save_folder, f"{entity_name}_{self.foot_segment_name}_{mode}_Average_KDE.csv"), index=False)
                
                axes[i].plot(time_range, average_kde)
                axes[i].set_title(f"Average KDE - {entity_name} \nMode: {mode}, {self.foot_segment_name}")
                axes[i].set_xlabel("Normalized cycle")
                axes[i].set_ylabel("Density")

                # Add beat positions as vertical lines
                for onst in [0, 0.25, 0.5, 0.75]:
                    axes[i].axvline(x=onst, color='g', linestyle='--', ymin=0.01, ymax=2)

                axes[i].minorticks_on()
                axes[i].grid(which='minor', alpha=0.2)
                axes[i].grid(which='major', alpha=0.5)
            else:
                axes[i].set_title(f"No data for {mode} mode in {entity_name}")


        # Save the figure
        fig.tight_layout()  # Adjust layout to prevent overlap
        save_path = os.path.join(save_folder, f"{entity_name}_{self.foot_segment_name}_Average_KDE_Subplot.png")
        fig.savefig(save_path, dpi=200)
        plt.close()
        print("Saving completed")

    # Function to compute and plot average KDE for all pieces
    def saveplot_all_pieces_kde(self, kde_data_all_pieces, save_path):
        modes = ['gr', 'in', 'au']  # The modes to iterate over
        time_range = np.linspace(0, 1, 1000)  # Example normalized cycle range
        
        
        # Create subplots for each mode (gr, in, au)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
        
        for i, mode in enumerate(modes):
            kde_list = kde_data_all_pieces[mode]
            if kde_list:  # Only proceed if there are KDE arrays for this mode
                average_kde = np.mean(kde_list, axis=0)
                
                print(f"Saving Average KDE all pieces...")
                df_avg_kde_all = pd.DataFrame(average_kde)
                df_avg_kde_all.to_csv(os.path.join(save_path, f"All_Pieces_Average_KDE_{mode}_{self.foot_segment_name}.csv"), index=False)
                
                
                # Plot for the combined subplot
                axes[i].plot(time_range, average_kde)
                axes[i].set_title(f"Average KDE for all pieces, Mode: {mode}, {self.foot_segment_name}")
                axes[i].set_xlabel("Normalized cycle")
                axes[i].set_ylabel('Density')
                for onset in [0, 0.25, 0.5, 0.75]:
                    axes[i].axvline(x=onset, color='g', linestyle='--', ymin=0.01, ymax=2)
                axes[i].minorticks_on()
                axes[i].grid(which='minor', alpha=0.2)
                axes[i].grid(which='major', alpha=0.5)
                
                # Create separate plots for each mode and save
                plt.figure(figsize=(8, 5), dpi=100)
                plt.plot(time_range, average_kde)
                plt.title(f"Average KDE for all pieces,  Mode: {mode}, {self.foot_segment_name}")
                plt.xlabel("Normalized cycle")
                plt.ylabel("Density")
                for onset in [0, 0.25, 0.5, 0.75]:
                    plt.axvline(x=onset, color='g', linestyle='--', ymin=0.01, ymax=2)
                plt.minorticks_on()
                plt.grid(which='minor', alpha=0.2)
                plt.grid(which='major', alpha=0.5)
                mode_save_path = os.path.join(save_path, f"All_Pieces_Average_KDE_Plot_{mode}_{self.foot_segment_name}.png")
                plt.tight_layout()
                plt.savefig(mode_save_path, dpi=200)
                plt.close()
                print(f"Separate plot saved at {save_path}")
        
    # Save the combined subplot
        
        fig.tight_layout()
        combined_save_path = os.path.join(save_path, f"All_Pieces_Average_KDE_Subplots_{self.foot_segment_name}.png")
        fig.savefig(combined_save_path, dpi=200)
        plt.close()
        print(f"Combined subplot saved at {combined_save_path}")



    def generate_foot_audio_click(self, Rankle_contact_binary, b_foot_cleaned_t, pos_thres, piece_folder_path, mocap_fps=240):
        
        print(f"Exporting Audio Click for {self.file_name}")
        click_duration = 50  # milliseconds
        click_freq = 1200  # Hz

        # Generate a single click sound
        click = Triangle(click_freq).to_audio_segment(duration=click_duration)

        onset_times = b_foot_cleaned_t
        dN = len(Rankle_contact_binary)  
        total_duration = (dN/mocap_fps)*1000  #  in milliseconds
        audio = AudioSegment.silent(duration=total_duration)
        
        # for onset in onset_times:
        #     position = int(onset * 1000)  # Convert onset time to milliseconds
        #     audio = audio.overlay(click, position=position)
        
        # Collect click overlays at each onset time
        overlays = [(click, int(onset * 1000)) for onset in b_foot_cleaned_t]
        # Overlay all clicks at once
        for click_segment, position in overlays:
            audio = audio.overlay(click_segment, position=position)
        
        
        # Export the audio with clicks to a file
        print(f"Exporting Audio Click for {self.file_name}")
        audio.export(os.path.join(piece_folder_path, f"{self.file_name}_{self.foot_segment_name}_OnsetsClick_{pos_thres}.wav"), format="wav")
        print(f"Exported successfully")
        
def create_folder(base_path, *subfolders):
    """
    Creates a folder at the specified base path, adding any subfolders provided.

    Parameters:
    - base_path: The base directory where the folders should be created.
    - subfolders: Additional subfolders (as a variable-length argument) to form the final path.
    """
    folder_path = os.path.join(base_path, *subfolders)  # Build the complete path using base and subfolders
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        pass
        # print(f"Folder '{os.path.join(*subfolders)}' already exists.")
    return folder_path

import os
import pandas as pd
import numpy as np

def save_onset_information(params, onsetinfo_folder_name):
    """
    Saves onset information to CSV files in a specified folder structure.
    
    Args:
        params (dict): Dictionary containing all necessary parameters:
            - base_logs_folder (str)
            - piece_folder_name (str)
            - file_name (str)
            - foot_segment_name (str)
            - both_foot_cleaned_t (array)
            - both_foot_cleaned (array)
            - rfoot_onsets (array)
            - lfoot_onsets (array)
            - rtoe_heel_diff (array)
            - rheel_toe_diff (array)
            - ltoe_heel_diff (array)
            - lheel_toe_diff (array)
            - onset_per_piece_info (list)
    """
    # Unpack dictionary values
    base_logs_folder = params["base_logs_folder"]
    piece_folder_name = params["piece_folder_name"]
    file_name = params["file_name"]
    foot_segment_name = params["foot_segment_name"]
    both_foot_cleaned_t = params["both_foot_cleaned_t"]
    both_foot_cleaned = params["both_foot_cleaned"]
    rfoot_onsets = params["rfoot_onsets"]
    lfoot_onsets = params["lfoot_onsets"]
    rtoe_heel_diff = params["rtoe_heel_diff"]
    rheel_toe_diff = params["rheel_toe_diff"]
    ltoe_heel_diff = params["ltoe_heel_diff"]
    lheel_toe_diff = params["lheel_toe_diff"]
    onset_per_piece_info = params["onset_per_piece_info"]

    # Create the output folder
    onsetinfo_folder_path = create_folder(base_logs_folder, piece_folder_name, onsetinfo_folder_name)
    
    # Record the number of footsteps per piece
    onset_per_piece_info.append((file_name, len(both_foot_cleaned_t)))
    
    # Define DataFrames for each data type
    df_both_feet = pd.DataFrame({"time_sec": both_foot_cleaned_t, "frames": both_foot_cleaned})
    df_rfoot_onsets = pd.DataFrame({"time_sec": rfoot_onsets / 240, "frames": rfoot_onsets})
    df_lfoot_onsets = pd.DataFrame({"time_sec": lfoot_onsets / 240, "frames": lfoot_onsets})
    df_rtoe_heel_timediff = pd.DataFrame({"time_sec": np.concatenate((rtoe_heel_diff / 240, rheel_toe_diff / 240)),
                                          "frames": np.concatenate((rtoe_heel_diff, rheel_toe_diff))})
    df_ltoe_heel_timediff = pd.DataFrame({"time_sec": np.concatenate((ltoe_heel_diff / 240, lheel_toe_diff / 240)),
                                          "frames": np.concatenate((ltoe_heel_diff, lheel_toe_diff))})
    
    # Save each DataFrame to CSV
    df_both_feet.to_csv(os.path.join(onsetinfo_folder_path, f"{file_name}_both_feet_onsets.csv"), index=False)  # both_feet_firstcontact_withminima_onsets
    df_rfoot_onsets.to_csv(os.path.join(onsetinfo_folder_path, f"{file_name}_right_foot_firstcontact_nominima_onsets.csv"), index=False)
    df_lfoot_onsets.to_csv(os.path.join(onsetinfo_folder_path, f"{file_name}_left_foot_firstcontact_nominima_onsets.csv"), index=False)
    df_rtoe_heel_timediff.to_csv(os.path.join(onsetinfo_folder_path, f"{file_name}_right_heel_toe_timediff.csv"), index=False)
    df_ltoe_heel_timediff.to_csv(os.path.join(onsetinfo_folder_path, f"{file_name}_left_heel_toe_timediff.csv"), index=False)

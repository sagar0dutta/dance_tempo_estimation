import os
import numpy as np
import matplotlib.pyplot as plt


class Inspect_plots:
    def __init__(self, file_name, mocap_fps, plot_folder_path, max_threshold_for_minima,
                 pos_diff_threshold_for_minima, pos_thres_for_ground, time_diff_threshold_for_minima, 
                 ank_toe_thres_sec):
        
        self.file_name = file_name
        self.mocap_fps = mocap_fps
        self.plot_folder_path = plot_folder_path
        
        self.max_threshold_for_minima = max_threshold_for_minima
        self.pos_diff_threshold_for_minima = pos_diff_threshold_for_minima
        self.time_diff_threshold_for_minima = time_diff_threshold_for_minima
        
        self.pos_thres_for_ground = pos_thres_for_ground
        self.ank_toe_thres_sec = ank_toe_thres_sec
        

    # inspect filtered minima
    def inspect_filtered_minima(self, Rshifted_ankle_pos_data, Rankle_maxima_filtered, Rankle_filtered_minima, Lshifted_ankle_pos_data, Lankle_maxima_filtered, Lankle_filtered_minima):
        
        dN = len(Lshifted_ankle_pos_data)
        dtime = np.arange(dN)/self.mocap_fps

        # Create the figure and subplots (2 rows, 1 column)
        f_filtered_min = plt.figure(figsize=(40,10), dpi=100)
        f_filtered_min.suptitle(f"Plot to inspect filtered minima of Right/Left Ankle\n"
                        f"pos_diff_threshold_for_minima: {self.pos_diff_threshold_for_minima}\n"
                        f"max_threshold_for_minima: {self.max_threshold_for_minima}\n"
                        f"time_diff_threshold_for_minima: {self.time_diff_threshold_for_minima}")

        # Left Ankle Data Subplot
        ax1 = f_filtered_min.add_subplot(3, 1, 1)  # 2 rows, 1 column, position 1
        ax1.plot(dtime, Lshifted_ankle_pos_data, linewidth=0.15, label='Lshifted Ankle Position')

        # Title and axes labels for the first plot
        ax1.set_title(f"{self.file_name} \n Left Ankle: Plot to inspect filtered minima Onsets")
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("Position (meters)")
        ax1.set_xticks(np.arange(0, dN/self.mocap_fps, 25))

        # Horizontal reference lines
        ax1.hlines(y=0.03, xmin=0, xmax=dN/self.mocap_fps, color='r', linestyle='dotted', linewidth=0.15)
        ax1.hlines(y=0.02, xmin=0, xmax=dN/self.mocap_fps, color='r', linestyle='dotted', linewidth=0.15)

        # Vertical event lines for maxima and minima
        for i, foot_event in enumerate(Lankle_maxima_filtered):
            ax1.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='g', linestyle='dotted', linewidth=0.15, label="Maxima (Green)" if i == 0 else None)
        for i, foot_event in enumerate(Lankle_filtered_minima):
            ax1.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.15, label="Minima (Blue)" if i == 0 else None)

        # Set grid and minor ticks
        ax1.minorticks_on()
        ax1.grid(which='minor', alpha=0.1)
        ax1.grid(which='major', alpha=0.5)
        ax1.legend(loc='upper left')

        # Right Ankle Data Subplot
        ax2 = f_filtered_min.add_subplot(3, 1, 2)  # 2 rows, 1 column, position 2
        ax2.plot(dtime, Rshifted_ankle_pos_data, linewidth=0.15, label='Rshifted Ankle Position')

        # Title and axes labels for the second plot
        ax2.set_title(f"{self.file_name} \n Right Ankle: Plot to inspect filtered minima Onsets")
        
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Position (meters)")
        ax2.set_xticks(np.arange(0, dN/self.mocap_fps, 25))

        # Horizontal reference lines
        ax2.hlines(y=0.05, xmin=0, xmax=dN/self.mocap_fps, color='r', linestyle='dotted', linewidth=0.15)
    

        # Vertical event lines for maxima and minima (Right Ankle, if available)
        for i, foot_event in enumerate(Rankle_maxima_filtered):
            ax2.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='g', linestyle='dotted', linewidth=0.15, label="Maxima (Green)" if i == 0 else None)
        for i, foot_event in enumerate(Rankle_filtered_minima):
            ax2.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.15, label="Minima (Blue)" if i == 0 else None)

        # Set grid and minor ticks
        ax2.minorticks_on()
        ax2.grid(which='minor', alpha=0.1)
        ax2.grid(which='major', alpha=0.5)
        ax2.legend(loc='upper left')

        # Third Subplot (ax3)
        ax3 = f_filtered_min.add_subplot(3, 1, 3)  # 3 rows, 1 column, position 3

        # Plot empty dotted line
        plot_empty = np.zeros(len(Rshifted_ankle_pos_data))
        ax3.plot(dtime, plot_empty, linewidth=0.2, color='r', linestyle="dotted")

        ax3.set_title(f"{self.file_name} \n Filtered Minima Onsets for Right (Red) Ankle and Left Ankle (Blue)")
        ax3.set_xlabel("Time (sec)")
        # ax3.set_ylabel("Minima Onsets")
        ax3.set_xticks(np.arange(0, dN/self.mocap_fps, 25))

        # Add vertical lines for right ankle minima
        for i, foot_event in enumerate(Rankle_filtered_minima):
            ax3.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='r', linestyle='dotted', linewidth=0.15, label="Right Ankle Minima (Red)" if i == 0 else None)

        # Add vertical lines for left ankle minima
        for i, foot_event in enumerate(Lankle_filtered_minima):
            ax3.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.15, label="Left Ankle Minima (Blue)" if i == 0 else None)
        ax3.legend(loc='upper left')


        # Show the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder_path, f"A_inspect_filtered_minima_onsets_{self.file_name}.png") , dpi = 600, format= 'png')
        plt.close()
        # plt.show()



    # Plot to inspect Ankle-Toe onsets

    def inspect_ground_contact_onsets(self, Rankle_contact_binary, Rtoe_contact_binary, rfoot_onsets, Lankle_contact_binary, Ltoe_contact_binary, lfoot_onsets):

        dN = len(Rankle_contact_binary)
        dtime = np.arange(dN) / self.mocap_fps

        f_binary = plt.figure(figsize=(50, 10), dpi=100)
        f_binary.suptitle(f"Plot to inspect ground contact onsets (ank_toe_thres_sec: {self.ank_toe_thres_sec})")


        ax1 = f_binary.add_subplot(3, 1, 1)  # 2 rows, 1 column, position 1
        ax1.plot(dtime, Rankle_contact_binary, linewidth=0.1, color='r', linestyle="dotted", label="Ankle (Red)")
        ax1.plot(dtime, Rtoe_contact_binary, linewidth=0.1, color='b', linestyle="dotted", label="Toe (Blue)")

        # Title and labels for the first subplot
        ax1.set_title(f"{self.file_name} \n Right Ankle & Toe Ground Contact Onsets")
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("Contact (Binary)")
        ax1.set_xticks(np.arange(0, dN/self.mocap_fps+25, 25))

        for i, foot_event in enumerate(rfoot_onsets):
            ax1.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='g', linestyle='--', linewidth=0.2, label="Selected onset (Green)" if i == 0 else None)
        ax1.legend(loc='upper left', fontsize= 14)

        ax2 = f_binary.add_subplot(3, 1, 2)  # 2 rows, 1 column, position 2
        ax2.plot(dtime, Lankle_contact_binary, linewidth=0.1, color='r', linestyle="dotted", label="Ankle (Blue)")
        ax2.plot(dtime, Ltoe_contact_binary, linewidth=0.1, color='b', linestyle="dotted", label="Toe (Red)")

        ax2.set_title(f"{self.file_name} \n Left Ankle & Toe Ground Contact Onsets")
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Contact (Binary)")
        ax2.set_xticks(np.arange(0, dN/self.mocap_fps+25, 25))

        for i, foot_event in enumerate(lfoot_onsets):
            ax2.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='g', linestyle='--', linewidth=0.2, label="Selected onset (Green)" if i == 0 else None)
        ax2.legend(loc='upper left', fontsize= 14)

        # Third Subplot (ax3)
        ax3 = f_binary.add_subplot(3, 1, 3)  # 3 rows, 1 column, position 3
        plot_empty = np.zeros(len(Rankle_contact_binary))
        ax3.plot(dtime, plot_empty, linewidth=0.2, color='r', linestyle="dotted")

        ax3.set_title("First Ground Contact of Right foot onsets (Red) and left foot onsets (Blue)")
        ax3.set_xlabel("Time (sec)")
        ax3.set_ylabel("Contact (Binary)")
        ax3.set_xticks(np.arange(0, dN/self.mocap_fps+25, 25))

        for foot_event in rfoot_onsets:
            ax3.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='r', linestyle='dotted', linewidth=0.25)
        for foot_event in lfoot_onsets:
            ax3.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.25)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder_path, f"B_inspect_ground_contact_onsets_{self.file_name}.png") , dpi = 600, format= 'png')
        plt.close()
        # plt.show()


    # Plot to inspect first_ground_contact and minima_onsets
    def inspect_first_ground_contact_and_minima_onsets(self, Rankle_contact_binary, rfoot_onsets, Rankle_filtered_minima, lfoot_onsets, Lankle_filtered_minima):

        dN = len(Rankle_contact_binary)
        dtime = np.arange(dN) / self.mocap_fps

        # Create figure
        f_foot_minima = plt.figure(figsize=(40, 10), dpi=100)

        # 1st Subplot (Right foot onsets)
        ax1 = f_foot_minima.add_subplot(2, 1, 1)  # 2 rows, 1 column, position 1
        plot_empty = np.zeros(len(Rankle_contact_binary))
        ax1.plot(dtime, plot_empty, linewidth=0.2, color='r', linestyle="dotted")

        ax1.set_title("Right Foot First Ground Contact (Red) and Filtered Minima (Blue)")
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("Contact (Binary)")
        ax1.set_xticks(np.arange(0, dN/self.mocap_fps+25, 25))

        # Right foot onsets and filtered minima
        for foot_event in rfoot_onsets:
            ax1.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='r', linestyle='dotted', linewidth=0.25)
        for foot_event in Rankle_filtered_minima:
            ax1.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.25)

        # 2nd Subplot (Left foot onsets)
        ax2 = f_foot_minima.add_subplot(2, 1, 2)  # 2 rows, 1 column, position 2
        ax2.plot(dtime, plot_empty, linewidth=0.2, color='r', linestyle="dotted")

        ax2.set_title("Left Foot First Ground Contact (Red) and Filtered Minima (Blue)")
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Contact (Binary)")
        ax2.set_xticks(np.arange(0, dN/self.mocap_fps+25, 25))

        # Left foot onsets and filtered minima
        for foot_event in lfoot_onsets:
            ax2.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='r', linestyle='dotted', linewidth=0.25)
        for foot_event in Lankle_filtered_minima:
            ax2.vlines(x=foot_event/self.mocap_fps, ymin=0.0, ymax=1, color='b', linestyle='dotted', linewidth=0.25)

        # Show the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_folder_path, f"C_inspect_first_ground_contact_and_minima_onsets{self.file_name}.png") , dpi = 600, format= 'png')
        plt.close()
        # plt.show()


    def inspect_first_ground_contact_and_filtered_minima(self, Rankle_contact_binary, 
                                rfoot_onsets, R_foot_minima_cleaned, 
                                lfoot_onsets, L_foot_minima_cleaned):
        
        dN = len(Rankle_contact_binary)
        dtime = np.arange(dN) / self.mocap_fps
        plot_empty = np.zeros(len(Rankle_contact_binary))
        
        f_filtered_foot_minima, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False, dpi=600)

        # Plot empty reference line in all subplots
        for ax in axes:
            ax.plot(dtime, plot_empty, linewidth=0.0)

        # Subplot 1: first ground contact and minimas in between
        axes[0].vlines(rfoot_onsets / self.mocap_fps, ymin=-0, ymax=1, color='r', linestyle='solid', label='R first ground contact', linewidth=0.1)
        axes[0].vlines(R_foot_minima_cleaned / self.mocap_fps, ymin=0, ymax=1, color='b', linestyle='dotted', label='R Minima filtered', linewidth=0.1)
        axes[0].legend(loc='upper left')
        
        # Subplot 2: first ground contact and minimas in between
        axes[1].vlines(lfoot_onsets / self.mocap_fps, ymin=-0, ymax=1, color='r', linestyle='solid', label='L first ground contact', linewidth=0.1)
        axes[1].vlines(L_foot_minima_cleaned / self.mocap_fps, ymin=0, ymax=1, color='b', linestyle='dotted', label='L Minima filtered', linewidth=0.1)
        axes[1].set_xlabel('Time (seconds)')
        axes[1].legend(loc='upper left')

        # General plot settings
        plt.suptitle("First Ground Contact and Filtered Minimas for Right and Left Foot")
        plt.tight_layout()  # Adjust layout to make room for the suptitle
        plt.savefig(os.path.join(self.plot_folder_path, f"D_inspect_first_ground_contact_and_filtered_minima_onsets_{self.file_name}.png") , dpi = 600, format= 'png')
        plt.close()
        # plt.show()
        
        
    def inspect_both_foot(self, Rankle_contact_binary, both_foot, both_foot_cleaned):
    
        dN = len(Rankle_contact_binary)
        dtime = np.arange(dN) / self.mocap_fps
        plot_empty = np.zeros(len(Rankle_contact_binary))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False, dpi=600)

        # Plot empty reference line in all subplots
        for ax in axes:
            ax.plot(dtime, plot_empty, linewidth=0.0)

        # Subplot 1: Both foot onsets
        axes[0].vlines(both_foot / self.mocap_fps, ymin=-0, ymax=1, color='b', linestyle='dotted', label='Both foot onsets', linewidth=0.1)
        axes[0].legend(loc='upper left')


        # Subplot 4: Both foot onsets Cleaned
        axes[1].vlines(both_foot_cleaned / self.mocap_fps, ymin=0, ymax=1, color='g', linestyle='dotted', label='Both foot onsets cleaned', linewidth=0.1)
        axes[1].set_xlabel('Time (seconds)')
        axes[1].legend(loc='upper left')


        plt.suptitle("Both foot onsets")
        plt.tight_layout()  # Adjust layout to make room for the suptitle
        plt.savefig(os.path.join(self.plot_folder_path, f"E_Inspect_Both_Foot_Onsets_{self.file_name}.png") , dpi = 600, format= 'png')
        plt.close()
        # plt.show()



















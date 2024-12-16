
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xsens.load_mvnx import load_mvnx

class djembe:
    def __init__(self, file_path):
        # self.file_path = file_path
        self.mvnx_file = load_mvnx(file_path)
        
        
    def load_djembe(self, frme):       # FRAMES_ALL = -1

        # mvnx_file = load_mvnx(file_path)
        
        # Segment names and IDs
        segment_names = [
            "SEGMENT_PELVIS", "SEGMENT_L5", "SEGMENT_L3", "SEGMENT_T12", "SEGMENT_T8",
            "SEGMENT_NECK", "SEGMENT_HEAD", 
            
            "SEGMENT_RIGHT_SHOULDER", "SEGMENT_RIGHT_UPPER_ARM",
            "SEGMENT_RIGHT_FOREARM", "SEGMENT_RIGHT_HAND", "SEGMENT_LEFT_SHOULDER",
            "SEGMENT_LEFT_UPPER_ARM", "SEGMENT_LEFT_FOREARM", "SEGMENT_LEFT_HAND",
            
            "SEGMENT_RIGHT_UPPER_LEG", "SEGMENT_RIGHT_LOWER_LEG", "SEGMENT_RIGHT_FOOT",
            "SEGMENT_RIGHT_TOE", "SEGMENT_LEFT_UPPER_LEG", "SEGMENT_LEFT_LOWER_LEG",
            "SEGMENT_LEFT_FOOT", "SEGMENT_LEFT_TOE"
        ]
        
        
        # joint_names = [
        # "JOINT_L5_S1", "JOINT_L4_L3", "JOINT_L1_T12", "JOINT_T9_T8", "JOINT_T1_C7", "JOINT_C1_HEAD",
        # "JOINT_RIGHT_T4_SHOULDER", "JOINT_RIGHT_SHOULDER", "JOINT_RIGHT_ELBOW", "JOINT_RIGHT_WRIST",
        # "JOINT_LEFT_T4_SHOULDER", "JOINT_LEFT_SHOULDER", "JOINT_LEFT_ELBOW", "JOINT_LEFT_WRIST",
        # "JOINT_RIGHT_HIP", "JOINT_RIGHT_KNEE", "JOINT_RIGHT_ANKLE", "JOINT_RIGHT_BALL_FOOT",
        # "JOINT_LEFT_HIP", "JOINT_LEFT_KNEE", "JOINT_LEFT_ANKLE", "JOINT_LEFT_BALL_FOOT"
        # ]
        
        # ergo_joint_names = ["ERGO_JOINT_T8_HEAD", "ERGO_JOINT_T8_LEFT_UPPER_ARM", "ERGO_JOINT_T8_RIGHT_UPPER_ARM", 
        #                     "ERGO_JOINT_PELVIS_T8", "ERGO_JOINT_VERTICAL_PELVIS", "ERGO_JOINT_VERTICAL_T8"]

        # joint_angle = {}
        # ergo_joint_angle = {}
        
        # for joint_id, joint_name in enumerate(joint_names):
        #     joint_angle[joint_name] = np.vstack(self.mvnx_file.get_joint_angle(joint_id, frame= frme))
        
        # for ergo_joint_id, ergo_joint_name in enumerate(ergo_joint_names):
        #     ergo_joint_angle[ergo_joint_name] = np.vstack(self.mvnx_file.get_ergo_joint_angle(ergo_joint_id, frame= frme))
        
        # Create dictionaries to hold data
        positions = {}
        velocities = {}
        accelerations = {}
        orientations = {}
        angular_velocities = {}
        angular_accelerations = {}
        sensor_orientation = {}
        
        # Iterate through each segment ID
        for segment_id, segment_name in enumerate(segment_names):
            positions[segment_name] = np.vstack(self.mvnx_file.get_segment_pos(segment_id, frame= frme))
            velocities[segment_name] = np.vstack(self.mvnx_file.get_segment_vel(segment_id, frame= frme))
            accelerations[segment_name] = np.vstack(self.mvnx_file.get_segment_acc(segment_id, frame= frme))
            
            orientations[segment_name] = np.vstack(self.mvnx_file.get_segment_ori(segment_id, frame= frme))       # The retrieval method will shift indices for w,x,y,z to 0,1,2,3
            angular_velocities[segment_name] = np.vstack(self.mvnx_file.get_segment_angular_vel(segment_id, frame= frme))
            angular_accelerations[segment_name] = np.vstack(self.mvnx_file.get_segment_angular_acc(segment_id, frame= frme))

            try:
                sensor_orientation[segment_name] = np.vstack(self.mvnx_file.get_sensor_ori(segment_id, frame= frme))
            except KeyError:
                # print("Not a sensor",segment_name)
                continue
                
        foot_contact =  self.mvnx_file.get_foot_contacts(frme)

        
        # foot_contacts = {"right_heel_contact": right_heel_contact,
        #                  "right_toe_contact": right_toe_contact,
        #                  "left_heel_contact": left_heel_contact,
        #                  "left_toe_contact": left_toe_contact}
        
        # FOOT_CONTACT_LEFT_HEEL = 1
        # FOOT_CONTACT_LEFT_TOE = 2
        # FOOT_CONTACT_RIGHT_HEEL = 4
        # FOOT_CONTACT_RIGHT_TOE = 8
        
        
        # joint_combinations = {
        #     "left_arm": ["JOINT_LEFT_SHOULDER", "JOINT_LEFT_ELBOW", "JOINT_LEFT_WRIST"],
            
        #     "right_arm": ["JOINT_RIGHT_SHOULDER", "JOINT_RIGHT_ELBOW", "JOINT_RIGHT_WRIST"],
            
        #     "left_leg": ["JOINT_LEFT_HIP", "JOINT_RIGHT_KNEE", "JOINT_RIGHT_ANKLE"],
            
        #     "right_leg": ["JOINT_RIGHT_HIP", "JOINT_RIGHT_KNEE", "JOINT_RIGHT_ANKLE"],
            
        #     "spine": ["JOINT_C1_HEAD", "JOINT_T1_C7", "JOINT_T9_T8", "JOINT_L1_T12", "JOINT_L4_L3", "JOINT_L5_S1"],
            
        # }
        
        
        segment_combinations = {
        "right_arm": ["SEGMENT_RIGHT_SHOULDER", "SEGMENT_RIGHT_UPPER_ARM", "SEGMENT_RIGHT_FOREARM", "SEGMENT_RIGHT_HAND"],
        "left_arm": ["SEGMENT_LEFT_SHOULDER", "SEGMENT_LEFT_UPPER_ARM", "SEGMENT_LEFT_FOREARM", "SEGMENT_LEFT_HAND"],
        
        
        "right_leg": ["SEGMENT_RIGHT_UPPER_LEG", "SEGMENT_RIGHT_LOWER_LEG", "SEGMENT_RIGHT_FOOT"],
        "left_leg": ["SEGMENT_LEFT_UPPER_LEG", "SEGMENT_LEFT_LOWER_LEG", "SEGMENT_LEFT_FOOT"],
        
        "spine": ["SEGMENT_HEAD", "SEGMENT_T8", "SEGMENT_L3", "SEGMENT_PELVIS"],
        
        "both_arm": ["SEGMENT_LEFT_SHOULDER", "SEGMENT_LEFT_UPPER_ARM", "SEGMENT_LEFT_FOREARM", "SEGMENT_LEFT_HAND", 
                    "SEGMENT_RIGHT_SHOULDER", "SEGMENT_RIGHT_UPPER_ARM", "SEGMENT_RIGHT_FOREARM", "SEGMENT_RIGHT_HAND"],
        
        "both_leg": ["SEGMENT_LEFT_UPPER_LEG", "SEGMENT_LEFT_LOWER_LEG", "SEGMENT_LEFT_FOOT", 
                    "SEGMENT_RIGHT_UPPER_LEG", "SEGMENT_RIGHT_LOWER_LEG", "SEGMENT_RIGHT_FOOT"],
        
        "all": ["SEGMENT_LEFT_SHOULDER", "SEGMENT_LEFT_UPPER_ARM", "SEGMENT_LEFT_FOREARM", "SEGMENT_LEFT_HAND", 
                "SEGMENT_RIGHT_SHOULDER", "SEGMENT_RIGHT_UPPER_ARM", "SEGMENT_RIGHT_FOREARM", "SEGMENT_RIGHT_HAND",
                "SEGMENT_LEFT_UPPER_LEG", "SEGMENT_LEFT_LOWER_LEG", "SEGMENT_LEFT_FOOT", 
                "SEGMENT_RIGHT_UPPER_LEG", "SEGMENT_RIGHT_LOWER_LEG", "SEGMENT_RIGHT_FOOT",
                "SEGMENT_HEAD", "SEGMENT_T8", "SEGMENT_L3", "SEGMENT_PELVIS"]
        }
        
            
        segments_pos = {}
        segments_vel = {}
        segments_acc = {}
        segments_ori = {}
        segments_ang_vel = {}
        segments_ang_acc = {}


        for segment, parts in segment_combinations.items():
            segments_pos[segment] = np.hstack([positions[part] for part in parts])
            segments_vel[segment] = np.hstack([velocities[part] for part in parts])
            segments_acc[segment] = np.hstack([accelerations[part] for part in parts])
            segments_ori[segment] = np.hstack([orientations[part] for part in parts])
            segments_ang_vel[segment] = np.hstack([angular_velocities[part] for part in parts])
            segments_ang_acc[segment] = np.hstack([angular_accelerations[part] for part in parts])
        
        
        joint_segment = {}
        
        
        
        # for jsegment, jparts in joint_combinations.items():
        #     joint_segment[jsegment] = np.hstack([joint_angle[part] for part in jparts])
            
        #     # Adds ergo joint angles to joint segments
        #     if jsegment == "right_arm":
        #         joint_segment[jsegment] = np.hstack([joint_segment[jsegment], ergo_joint_angle["ERGO_JOINT_T8_RIGHT_UPPER_ARM"]])
        
        #     if jsegment == "left_arm":
        #         joint_segment[jsegment] = np.hstack([joint_segment[jsegment], ergo_joint_angle["ERGO_JOINT_T8_LEFT_UPPER_ARM"]])

        #     if jsegment == "spine":
        #         joint_segment[jsegment] = np.hstack([joint_segment[jsegment], ergo_joint_angle["ERGO_JOINT_T8_HEAD"], ergo_joint_angle["ERGO_JOINT_VERTICAL_PELVIS"]])
        
        




        segment_label = {
            "right_leg": [
                "x_right_upper_leg", "y_right_upper_leg", "z_right_upper_leg",
                "x_right_lower_leg", "y_right_lower_leg", "z_right_lower_leg",
                "x_right_foot", "y_right_foot", "z_right_foot"
            ],
            
            "left_leg": [
                "x_left_upper_leg", "y_left_upper_leg", "z_left_upper_leg",
                "x_left_lower_leg", "y_left_lower_leg", "z_left_lower_leg",
                "x_left_foot", "y_left_foot", "z_left_foot"
            ],
            
            "left_arm": [
                "x_left_shoulder", "y_left_shoulder", "z_left_shoulder",
                "x_left_upper_arm", "y_left_upper_arm", "z_left_upper_arm",
                "x_left_forearm", "y_left_forearm", "z_left_forearm",
                "x_left_hand", "y_left_hand", "z_left_hand"
            ],
            
            "right_arm": [
                "x_right_shoulder", "y_right_shoulder", "z_right_shoulder",
                "x_right_upper_arm", "y_right_upper_arm", "z_right_upper_arm",
                "x_right_forearm", "y_right_forearm", "z_right_forearm",
                "x_right_hand", "y_right_hand", "z_right_hand"
            ],
            
            "spine": [
                "x_head", "y_head", "z_head",
                "x_t8", "y_t8", "z_t8",
                "x_l3", "y_l3", "z_l3",
                "x_pelvis", "y_pelvis", "z_pelvis"
            ],
            
            "both_arm": [
                "x_left_shoulder", "y_left_shoulder", "z_left_shoulder",
                "x_left_upper_arm", "y_left_upper_arm", "z_left_upper_arm",
                "x_left_forearm", "y_left_forearm", "z_left_forearm",
                "x_left_hand", "y_left_hand", "z_left_hand",
                "x_right_shoulder", "y_right_shoulder", "z_right_shoulder",
                "x_right_upper_arm", "y_right_upper_arm", "z_right_upper_arm",
                "x_right_forearm", "y_right_forearm", "z_right_forearm",
                "x_right_hand", "y_right_hand", "z_right_hand"
            ],
            
            "both_leg": [
                "x_left_upper_leg", "y_left_upper_leg", "z_left_upper_leg",
                "x_left_lower_leg", "y_left_lower_leg", "z_left_lower_leg",
                "x_left_foot", "y_left_foot", "z_left_foot",
                "x_right_upper_leg", "y_right_upper_leg", "z_right_upper_leg",
                "x_right_lower_leg", "y_right_lower_leg", "z_right_lower_leg",
                "x_right_foot", "y_right_foot", "z_right_foot"
            ],
            
            "all": [
                "x_left_shoulder", "y_left_shoulder", "z_left_shoulder",
                "x_left_upper_arm", "y_left_upper_arm", "z_left_upper_arm",
                "x_left_forearm", "y_left_forearm", "z_left_forearm",
                "x_left_hand", "y_left_hand", "z_left_hand",
                "x_right_shoulder", "y_right_shoulder", "z_right_shoulder",
                "x_right_upper_arm", "y_right_upper_arm", "z_right_upper_arm",
                "x_right_forearm", "y_right_forearm", "z_right_forearm",
                "x_right_hand", "y_right_hand", "z_right_hand",
                "x_left_upper_leg", "y_left_upper_leg", "z_left_upper_leg",
                "x_left_lower_leg", "y_left_lower_leg", "z_left_lower_leg",
                "x_left_foot", "y_left_foot", "z_left_foot",
                "x_right_upper_leg", "y_right_upper_leg", "z_right_upper_leg",
                "x_right_lower_leg", "y_right_lower_leg", "z_right_lower_leg",
                "x_right_foot", "y_right_foot", "z_right_foot",
                "x_head", "y_head", "z_head",
                "x_t8", "y_t8", "z_t8",
                "x_l3", "y_l3", "z_l3",
                "x_pelvis", "y_pelvis", "z_pelvis"
            ]
        }
        
        
        segments_all = {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "orientation": orientations,
            "angular_velocity": angular_velocities,
            "angular_accelaration": angular_accelerations,
            "sensor_orientation": sensor_orientation,
            "foot_contacts": foot_contact,
        }
        
        composite_segments = {
            "position": segments_pos,
            "velocity": segments_vel,
            "accelaration": segments_acc,
            "orientation": segments_ori,
            "angular_velocity": segments_ang_vel,
            "angular_accelaration": segments_ang_acc,
        }
        

        return segments_all, composite_segments, joint_segment, segment_label

def calculate_pca(data_normalized, nb_pca):
    cov_matrix = np.cov(data_normalized.T)                                  # Step 2: Covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)                    #  Eigendecomposition

    sorted_indices = np.argsort(eigenvalues)[::-1]                          # Sort eigenvalues in descending order and select the top k eigenvectors
    selected_eigenvectors = eigenvectors[:, sorted_indices[:nb_pca]]

    data_reduced = np.dot(data_normalized, selected_eigenvectors)       # Dimensionality reduction

    pca_result = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "data_reduced": data_reduced,
        "cov_matrix": cov_matrix,
        "selected_eigenvectors": selected_eigenvectors,
    }
    
    return pca_result

def plot_variance_heat(eigenvalues, selected_eigenvectors, cov_matrix, y_axis_labels):
    
    # Plot cumulative variance
    tot = sum(eigenvalues)
    var_exp = [(i / tot)*100 for i in sorted(eigenvalues, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)


    fig_heat = plt.figure(figsize=(10, 8))
    sns.heatmap(selected_eigenvectors, annot = True, cmap="PiYG", yticklabels=y_axis_labels)       #, yticklabels=y_axis_labels
    plt.title("Heatmap of PC or Eigenmovements")
    plt.show() 


    
    fig_var = plt.figure(figsize=(10, 8))
    with plt.style.context('seaborn-v0_8-whitegrid'):

        plt.bar(range(cov_matrix.shape[0]), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(cov_matrix.shape[0]), cum_var_exp, where='mid',
                label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.title("Variance of PC's")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
    return fig_heat, fig_var



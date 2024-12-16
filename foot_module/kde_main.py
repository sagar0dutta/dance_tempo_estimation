from foot_module import onset_calculations, utils

OnsetProcessor = onset_calculations.OnsetProcessor()


def process_onsets(data):
    kde_list = []  # collect feet_onsets for the 3 modes for subplot

    # Unpack data dictionary
    modes = data['modes']
    df_annotation = data['df_annotation']
    loaded_mcycle_onsets = data['loaded_mcycle_onsets']
    base_logs_folder = data['base_logs_folder']
    piece_folder_name = data['piece_folder_name']
    file_name = data['file_name']
    foot_segment_name = data['foot_segment_name']
    kde_folder_path = data['kde_folder_path']
    kde_data_all_pieces = data['kde_data_all_pieces']
    kde_data_per_piece = data['kde_data_per_piece']
    piece_name = data['piece_name']
    ensemble_name = data['ensemble_name']
    day = data['day']
    kde_data_per_dancer = data['kde_data_per_dancer']
    both_foot_cleaned_t = data['both_foot_cleaned_t']
    Rankle_contact_binary = data['Rankle_contact_binary']
    pos_thres_for_ground = data['pos_thres_for_ground']
    piece_folder_path = data['piece_folder_path']
    
    for mocap_mode in modes:

        category_df = df_annotation.groupby('mocap')
        category_df = category_df.get_group(mocap_mode)
        category_df = category_df.reset_index(drop=True)

        choose_nb_onset_to_make_block = 2
        section_data = OnsetProcessor.onset_calculations(category_df, loaded_mcycle_onsets, choose_nb_onset_to_make_block)

        # directoru creation
        mode_folder_path = utils.create_folder(base_logs_folder, piece_folder_name, mocap_mode)
        
        utils_func = utils.Utils_func(section_data, file_name, foot_segment_name, mocap_mode, mode_folder_path, kde_folder_path)
        feet_onsets = utils_func.collect_foot_onsets_per_cycle(both_foot_cleaned_t)
        
        kde = utils_func.kde_plot_save(feet_onsets)     # average kde
        kde_list.append(kde)    # for subplotting the 3 modes
        
        kde_data_all_pieces[mocap_mode].append(kde)     # kde across all pieces
        kde_data_per_piece[piece_name][mocap_mode].append(kde)  # kde per piece type

        if (ensemble_name == "E1" and day == "D1") or (ensemble_name == "E1" and day == "D5"):
            kde_data_per_dancer["dancer_1"][mocap_mode].append(kde)
        if (ensemble_name == "E1" and day == "D2"):
            kde_data_per_dancer["dancer_2"][mocap_mode].append(kde)
        if (ensemble_name == "E2" and day == "D3") or (ensemble_name == "E2" and day == "D4"):
            kde_data_per_dancer["dancer_3"][mocap_mode].append(kde)
        if (ensemble_name == "E3" and day == "D5") or (ensemble_name == "E3" and day == "D6"):
            kde_data_per_dancer["dancer_4"][mocap_mode].append(kde)


    utils_func.kde_subplot_save(kde_list, modes, kde_folder_path)    # for subplotting the 3 modes kde for each piece
    utils_func.generate_foot_audio_click(Rankle_contact_binary, both_foot_cleaned_t, pos_thres_for_ground, piece_folder_path)
    
    return kde_data_all_pieces, kde_data_per_piece, kde_data_per_dancer


data_dict = {
    'modes': modes,
    'df_annotation': df_annotation,
    'loaded_mcycle_onsets': loaded_mcycle_onsets,
    'base_logs_folder': base_logs_folder,
    'piece_folder_name': piece_folder_name,
    'file_name': file_name,
    'foot_segment_name': foot_segment_name,
    'kde_folder_path': kde_folder_path,
    'kde_data_all_pieces': kde_data_all_pieces,
    'kde_data_per_piece': kde_data_per_piece,
    'piece_name': piece_name,
    'ensemble_name': ensemble_name,
    'day': day,
    'kde_data_per_dancer': kde_data_per_dancer,
    'both_foot_cleaned_t': both_foot_cleaned_t,
    'Rankle_contact_binary': Rankle_contact_binary,
    'pos_thres_for_ground': pos_thres_for_ground,
    'piece_folder_path': piece_folder_path
    }

    kde_data_all_pieces, kde_data_per_piece, kde_data_per_dancer = process_onsets(data_dict)
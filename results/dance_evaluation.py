import numpy as np



def evaluate_dance_onsets(drum_onsets, dance_onsets, tolerance=0.1):
    matched_time_diffs = []
    matched_indices = []
    unmatched_indices = []
    
    for d in dance_onsets:
        # Find the nearest drum onset
        time_diffs = np.abs(drum_onsets - d)
        min_diff = np.min(time_diffs)
        min_index = np.argmin(time_diffs)
        
        # Check if within tolerance
        if min_diff <= tolerance:
            matched_time_diffs.append(min_diff)
            matched_indices.append(min_index)
        else:
            # Unmatched dance onset
            unmatched_indices.append(min_index)  # collect unmatched onsets
    
    # Compute metrics
    num_matched = len(matched_time_diffs)
    num_unmatched = len(unmatched_indices)
    total_dance_onsets = len(dance_onsets)
    matching_rate = num_matched / total_dance_onsets if total_dance_onsets > 0 else 0
    average_time_diff = np.mean(matched_time_diffs) if matched_time_diffs else None
    
    print("total onsets:", len(dance_onsets))
    print("number of match:", num_matched)
    print("number of mis-match:", num_unmatched)
    print("matching rate:", matching_rate)
    print("average time diff:", average_time_diff)
    
    results = {
        "Total Onsets": len(dance_onsets),
        "Number of Match": num_matched,
        "Number of Match": num_matched,
        "Number of Mis-match": num_unmatched,
        "Matching Rate": matching_rate,
        "Average Time Diff": average_time_diff,
    }
    
    return results

def evaluate_dance_onsets_with_half_beats(drum_onsets, dance_onsets, tolerance=0.05):
    matched_time_diffs_primary = []
    matched_time_diffs_secondary = []
    unmatched_dance_onsets = []

    drum_onsets = np.array(drum_onsets)
    dance_onsets = np.array(dance_onsets)
    P_match_count = 0 
    S_match_count = 0
    for d in dance_onsets:
        # Primary Matching: Find the nearest drum onset
        time_diffs = np.abs(drum_onsets - d)
        min_diff = np.min(time_diffs)
        min_index = np.argmin(time_diffs)
        
        if min_diff <= tolerance:
            # Primary match found
            matched_time_diffs_primary.append(min_diff)
            P_match_count += 1
        else:
            # Unmatched dance onset, proceed to secondary matching
            unmatched_dance_onsets.append(d)
    
    # Prepare for secondary matching
    drum_onsets_sorted = np.sort(drum_onsets)
    unmatched_dance_onsets = np.array(unmatched_dance_onsets)

    for d_u in unmatched_dance_onsets:
        # Find preceding and following drum onsets
        preceding = drum_onsets_sorted[drum_onsets_sorted < d_u]
        following = drum_onsets_sorted[drum_onsets_sorted > d_u]
        
        if len(preceding) == 0 or len(following) == 0:
            # Cannot find a valid half-beat (dance onset is outside the range of drum onsets)
            continue
        
        b_p = preceding[-1]
        b_f = following[0]
        
        # Calculate half-beat
        h = (b_p + b_f) / 2.0
        
        # Secondary Matching: Check if dance onset matches the half-beat
        if np.abs(d_u - h) <= tolerance:
            # Secondary match found
            matched_time_diffs_secondary.append(np.abs(d_u - h))
            S_match_count += 1
        else:
            # Remains unmatched
            pass  # Optionally collect unmatched dance onsets for further analysis

    # Compute metrics
    N_dance = len(dance_onsets)
    N_primary = len(matched_time_diffs_primary)
    N_secondary = len(matched_time_diffs_secondary)
    N_unmatched = N_dance - N_primary - N_secondary

    PMR = N_primary / N_dance if N_dance > 0 else 0
    SMR = N_secondary / N_dance if N_dance > 0 else 0
    OMR = (N_primary + N_secondary) / N_dance if N_dance > 0 else 0

    ATD_primary = np.mean(matched_time_diffs_primary) if matched_time_diffs_primary else None
    ATD_secondary = np.mean(matched_time_diffs_secondary) if matched_time_diffs_secondary else None

    
    
    results = {
        "Total Onsets": len(dance_onsets),
        "Total Primary Match": P_match_count,
        "Total Secondary Match": S_match_count,
        'Primary Matching Rate': PMR,
        'Secondary Matching Rate': SMR,
        'Overall Matching Rate': OMR,
        'Average Time Difference (Primary)': ATD_primary,
        'Average Time Difference (Secondary)': ATD_secondary,
        'Number of Unmatched Dance Onsets': N_unmatched
    }

    return results


def calculate_metrics_with_oe(ref, calculated, tolerance=5):
    """
    Calculate Acc1, Acc2, Metric3, OE1, and OE2.
    Args:
        ref (np.ndarray): Reference BPMs.
        calculated (np.ndarray): Estimated BPMs.
        tolerance (float): Precision window (default 4%).
    Returns:
        dict: Dictionary with Acc1, Acc2, Metric3, OE1, and OE2 values.
    """
    # ref_tolerance = ref * tolerance
    ref_tolerance = tolerance
    
    # Acc1: Within 4% of reference BPM
    acc1_count = np.sum(np.abs(calculated - ref) <= ref_tolerance)
    
    # Acc2: Within 4% of reference BPM, double, or half
    acc2_count = np.sum(
        # (np.abs(calculated - ref) <= ref_tolerance) |
        (np.abs(calculated - 2 * ref) <= ref_tolerance) |
        (np.abs(calculated - ref / 2) <= ref_tolerance)
    )
    
    error_half = np.abs(calculated - ref / 2)
    # error_1x = np.abs(calculated - ref)
    error_2x = np.abs(calculated - 2 * ref)
    
    scales = [1, 2, 0.5, 3, 1/3]

    # OE1: Overestimated BPM outside hierarchical relationships
    oe1_count = np.sum(
        (calculated > ref) &  # Overestimation
        ~np.any([np.abs(calculated - ref * scale) <= ref_tolerance for scale in scales], axis=0)  # Not within any scale
    )
    
    # OE2: Overestimated BPM within hierarchical relationships
    oe2_count = np.sum(
        (calculated > ref) &  # Overestimation
        np.any([np.abs(calculated - ref * scale) <= ref_tolerance for scale in scales], axis=0)  # Within any scale
    )
    
    total = len(ref)
    
    metrics = {
        "acc1": (acc1_count / total) * 100,
        "acc2": (acc2_count / total) * 100,
        "OE1": (oe1_count / total) * 100,
        "OE2": (oe2_count / total) * 100,
    }
    return metrics


# def calculate_metrics_with_oe(ref, calculated, tolerance=5):
#     """
#     Calculate Acc1, Acc2, OE1, OE2, AOE1, and AOE2 metrics.

#     Args:
#         ref (np.ndarray): Reference BPMs.
#         calculated (np.ndarray): Estimated BPMs.
#         tolerance (float): Precision window (default 5 BPM).

#     Returns:
#         dict: Dictionary with Acc1, Acc2, OE1, OE2, AOE1, and AOE2 values.
#     """
#     ref = np.array(ref)
#     calculated = np.array(calculated)
#     scales = [1, 2, 0.5, 3, 1/3]  # Scales for hierarchical relationships
    
#     # Ensure matching lengths
#     assert len(ref) == len(calculated), "Reference and calculated arrays must have the same length."
    
#     # 1. Acc1: Within tolerance of reference BPM
#     acc1_count = np.sum(np.abs(calculated - ref) <= tolerance)
    
#     # 2. Acc2: Within tolerance of reference BPM, double, or half
#     acc2_count = np.sum(
#         (np.abs(calculated - ref) <= tolerance) |
#         (np.abs(calculated - 2 * ref) <= tolerance) |
#         (np.abs(calculated - ref / 2) <= tolerance)
#     )
    
#     # 3. OE1: Octave error (log2 of ratio)
#     oe1 = np.log2(calculated / ref)
    
#     # 4. OE2: Min absolute octave error across scales
#     all_scaled_errors = [np.log2(scale * calculated / ref) for scale in scales]
#     oe2 = np.min(np.abs(all_scaled_errors), axis=0)  # Smallest absolute error across scales

#     # 5. Absolute OE metrics
#     aoe1 = np.abs(oe1)  # Absolute OE1
#     aoe2 = np.abs(oe2)  # Absolute OE2

#     # Normalize metrics for percentage
#     total = len(ref)
    
#     metrics = {
#         "Acc1": (acc1_count / total) * 100,
#         "Acc2": (acc2_count / total) * 100,
#         "OE1": oe1,
#         "OE2": oe2,
#         "AOE1": np.mean(aoe1),  # Mean absolute OE1
#         "AOE2": np.mean(aoe2)   # Mean absolute OE2
#     }
#     return metrics
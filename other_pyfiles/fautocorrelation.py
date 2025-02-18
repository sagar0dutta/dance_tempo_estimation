import numpy as np
from scipy.signal import find_peaks

def binary_to_peak(binary_array, sampling_rate=240, peak_duration=0.05):
    """
    Represent binary array with 50ms bandwidth peaks.

    Parameters:
        binary_array (numpy.ndarray): Binary input array (1s and 0s).
        sampling_rate (int): Sampling rate in Hz.
        peak_duration (float): Duration of each peak in seconds (default is 0.05s).

    Returns:
        numpy.ndarray: Continuous signal with peaks for each 1 in the binary array.
    """
    n = len(binary_array)
    peak_samples = int(peak_duration * sampling_rate)  # Samples in 50ms
    half_peak = peak_samples // 2

    # Create a Gaussian peak
    t = np.linspace(-half_peak, half_peak, peak_samples)
    peak_shape = np.exp(-0.5 * (t / (half_peak / 2))**2)  # Gaussian

    # Create the continuous signal
    continuous_signal = np.zeros(n + peak_samples)
    for i, value in enumerate(binary_array):
        if value == 1:
            start = i
            end = i + peak_samples
            continuous_signal[start:end] += peak_shape

    # Trim to original length
    return continuous_signal[:n]


def binary_autocorrelation(data):
    """
    Compute autocorrelation for a binary numpy array.

    Parameters:
        data (numpy.ndarray): Binary input array (1s and 0s).

    Returns:
        numpy.ndarray: Autocorrelation values.
    """
    n = len(data)
    mean = sum(data) / n
    centered_data = data - mean  # Center the binary array
    variance = sum(centered_data ** 2)

    if variance == 0:
        # If all values are the same, return 0 for all lags
        return np.zeros(n)

    autocorr = []
    for lag in range(n):
        cov = sum(centered_data[i] * centered_data[i + lag] for i in range(n - lag))
        autocorr.append(cov / variance)

    return np.array(autocorr)


def find_top_peaks_scipy(autocorr_values, num_peaks=3):
    """
    Find the top N peaks in an autocorrelation array using scipy.

    Parameters:
        autocorr_values (numpy.ndarray): Autocorrelation values.
        num_peaks (int): Number of top peaks to retrieve.

    Returns:
        list of tuples: Each tuple contains (lag, peak_value) sorted by peak_value.
    """
    # Find all peaks
    peaks, _ = find_peaks(autocorr_values)

    # Extract peak values and their lags
    peak_values = autocorr_values[peaks]
    peak_lags = peaks

    # Sort peaks by value in descending order
    sorted_indices = np.argsort(peak_values)[::-1]
    top_indices = sorted_indices[:num_peaks]

    # Return top N peaks as (lag, peak_value) tuples
    return [(peak_lags[i], peak_values[i]) for i in top_indices]






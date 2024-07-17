from __future__ import print_function
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import time

def pitch_autocorrelation(frame, sample_rate, min_lag, max_lag):
    corr = fftconvolve(frame, frame[::-1], mode='full')
    corr = corr[len(corr)//2:len(corr)//2 + max_lag]
    peak = min_lag + np.argmax(corr[min_lag:])
    if corr[peak] > 0.5 * np.max(corr):
        return sample_rate / float(peak)
    return 0

def pitch_autocorrelation_2(frames, sample_rate, min_lag, max_lag):
    flipped = np.flip(frames, axis=1)
    corr = fftconvolve(frames, flipped, mode='full', axes=1)
    corr_len = len(corr[0])
    corr = corr[:, corr_len//2:corr_len//2 + max_lag]
    peak_indices = min_lag + np.argmax(corr[:, min_lag:], axis=1)
    peaks = np.max(corr[:, min_lag:], axis=1)
    good_peaks = peaks > 0.5 * np.max(corr, axis=1)
    return float(sample_rate) / peak_indices[good_peaks]

def analyze_sound_2(file_path):
    sr, y = wavfile.read(file_path)
    y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # Intensity analysis
    frame_length = 2048
    hop_length = 1024
    n_frames = 1 + (len(y) - frame_length) // hop_length
    rms = np.sqrt(np.mean(np.square(y[:n_frames*hop_length].reshape(-1, hop_length)), axis=1))
    db_rms = 20 * np.log10(np.maximum(rms, 1e-5)/0.00002)
    mean_intensity = np.mean(db_rms)
    std_intensity = np.std(db_rms)

    # Pitch analysis
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    min_freq, max_freq = 75, 300
    min_lag, max_lag = int(sr / max_freq), int(sr / min_freq)
    n_frames = (len(y) - frame_length) // hop_length

    shape=(n_frames, frame_length)

    # Note: Stride is expressed in bytes
    element_size = y.strides[0]
    strides=(hop_length * element_size, element_size)

    # See documentation for this function; we need to be careful
    # to stay within the bounds of the numpy array.
    strided_y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

    pitches = pitch_autocorrelation_2(strided_y, sr, min_lag, max_lag)

    mean_pitch = np.mean(pitches)
    std_pitch = np.std(pitches)

    min_amplitude, max_amplitude = np.min(y), np.max(y)

    return (mean_intensity, std_intensity, mean_pitch, std_pitch, min_amplitude, max_amplitude)


def analyze_sound(file_path):
    sr, y = wavfile.read(file_path)
    y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # Intensity analysis
    frame_length = 2048
    hop_length = 1024
    n_frames = 1 + (len(y) - frame_length) // hop_length
    rms = np.sqrt(np.mean(np.square(y[:n_frames*hop_length].reshape(-1, hop_length)), axis=1))
    db_rms = 20 * np.log10(np.maximum(rms, 1e-5)/0.00002)
    mean_intensity = np.mean(db_rms)
    std_intensity = np.std(db_rms)

    # Pitch analysis
    frame_length = int(0.04 * sr)
    hop_length = int(0.01 * sr)
    min_freq, max_freq = 75, 300
    min_lag, max_lag = int(sr / max_freq), int(sr / min_freq)
    n_frames = (len(y) - frame_length) // hop_length
    pitches = []
    for i in range(0, n_frames * hop_length, hop_length):
        pitch = pitch_autocorrelation(y[i:i+frame_length], sr, min_lag, max_lag)
        if pitch != 0:
            pitches.append(pitch)

    mean_pitch = np.mean(pitches) if pitches else 0
    std_pitch = np.std(pitches) if pitches else 0

    min_amplitude, max_amplitude = np.min(y), np.max(y)

    return (mean_intensity, std_intensity, mean_pitch, std_pitch, min_amplitude, max_amplitude)

if __name__ == "__main__":
    file_path = "./pause_10.wav"
    start_time = time.time()
    results = analyze_sound_2(file_path)
    end_time = time.time()

    print("File: {}".format(file_path))
    print("Mean Intensity (dB): {:.6f}".format(results[0]))
    print("Intensity Std Dev: {:.6f}".format(results[1]))
    print("Mean Pitch (Hz): {:.6f}".format(results[2]))
    print("Pitch Std Dev (Hz): {:.6f}".format(results[3]))
    print("Min Amplitude: {:.6f}".format(results[4]))
    print("Max Amplitude: {:.6f}".format(results[5]))
    print("Execution time: {:.3f} seconds".format(end_time - start_time))

import librosa
import numpy as np


def time_masking(wav, t):
    num_frames = wav.shape[0]
    assert t <= num_frames
    start = np.random.randint(0, num_frames - t)
    end = start + t
    wav[start:end] = np.min(wav)

    return wav


def frequency_masking(wav, f):
    num_freqs = wav.shape[1]
    assert f <= num_freqs


def time_warping(wav):
    print()

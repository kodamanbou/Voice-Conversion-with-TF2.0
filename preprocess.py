import librosa
import glob
import hyperparameter as hp
import pyworld
import numpy as np
import pickle
from tqdm import tqdm


def load_wavs(wav_dir):
    wavs = []
    for file in glob.glob(wav_dir + '/*.wav'):
        wav, _ = librosa.load(file, sr=hp.rate)
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=800.0, frame_period=hp.duration)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-cepstral coefficients (MCEPs)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in tqdm(wavs):
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    # decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    return wav_padded


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted


if __name__ == '__main__':
    wavs_A = load_wavs('./datasets/my_voice')
    wavs_B = load_wavs('./datasets/target_voice')

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs_A, fs=hp.rate, coded_dim=hp.num_mceps)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs_B, fs=hp.rate, coded_dim=hp.num_mceps)

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_B_transposed)

    print('Saving...')
    with open('./datasets/my_voice/my_voice.p', 'wb') as f:
        pickle.dump((coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A), f)

    with open('./datasets/target_voice/target_voice.p', 'wb') as f:
        pickle.dump((coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B), f)

    print('Done')

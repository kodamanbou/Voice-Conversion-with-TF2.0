import tensorflow as tf
import numpy as np
import pickle
import librosa
from librosa import display
import matplotlib.pyplot as plt
from model import CycleGAN2
import hyperparameter as hp
from preprocess import wav_padding, pitch_conversion, world_speech_synthesis
from preprocess import world_decompose, world_encode_spectral_envelop, world_decode_spectral_envelop


def seg_and_pad(src, n_frames):
    n_origin = src.shape[1]
    n_padded = (n_origin // n_frames + 1) * n_frames
    left_pad = (n_padded - n_origin) // 2
    right_pad = n_padded - n_origin - left_pad
    src = np.pad(src, [(0, 0), (left_pad, right_pad)], 'constant', constant_values=0)
    src = np.reshape(src, [-1, hp.num_mceps, n_frames])

    return src


model = CycleGAN2()
latest = tf.train.latest_checkpoint(hp.logdir)
model.load_weights(latest)

print('Loading cached data...')
with open('./datasets/my_voice/my_voice.p', 'rb') as f:
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = pickle.load(f)

with open('./datasets/target_voice/target_voice.p', 'rb') as f:
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = pickle.load(f)

wav, _ = librosa.load('./datasets/target_voice/voice28.wav', sr=hp.rate)
wav = wav_padding(wav, hp.rate, hp.duration)
f0, timeaxis, sp, ap = world_decompose(wav, hp.rate)
f0_converted = pitch_conversion(f0, log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B)
coded_sp = world_encode_spectral_envelop(sp, hp.rate, hp.num_mceps)
coded_sp_transposed = coded_sp.T
coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
coded_sp_norm = seg_and_pad(coded_sp_norm, hp.n_frames)

wav_forms = []
for sp_norm in coded_sp_norm:
    sp_norm = np.expand_dims(sp_norm, axis=-1)
    coded_sp_converted_norm = model([sp_norm, sp_norm])[0][0]
    if coded_sp_converted_norm.shape[1] > len(f0):
        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
    elif coded_sp_converted_norm.shape[1] < len(f0):
        f0_converted = f0_converted[:coded_sp_converted_norm.shape[1]]
        ap = ap[:coded_sp_converted_norm.shape[1]]

    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
    coded_sp_converted = np.array(coded_sp_converted, dtype=np.float64).T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decode_sp_converted = world_decode_spectral_envelop(coded_sp_converted, hp.rate)
    wav_transformed = world_speech_synthesis(f0_converted, decode_sp_converted, ap, hp.rate, hp.duration)
    wav_forms.append(wav_transformed)

wav_forms = np.concatenate(wav_forms)
librosa.output.write_wav('./outputs/test.wav', wav_forms, hp.rate)

test_wav, _ = librosa.load('./outputs/test.wav', sr=hp.rate)
D = librosa.amplitude_to_db(np.abs(librosa.stft(test_wav)), ref=np.max)
plt.figure(figsize=(12, 8))
display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()

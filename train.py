import tensorflow as tf
import numpy as np
import librosa
import pickle
import os
import glob
import datetime
import hyperparameter as hp
from preprocess import world_decompose, pitch_conversion, world_encode_spectral_envelop, world_decode_spectral_envelop, \
    world_speech_synthesis
from utils import l1_loss, l2_loss
from model import CycleGAN2


@tf.function
def train_step(inputs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        outputs = model(inputs)
        generation_A = outputs[0]
        generation_B = outputs[1]
        cycle_A = outputs[2]
        cycle_B = outputs[3]
        identity_A = outputs[4]
        identity_B = outputs[5]
        discrimination_A_real = outputs[6]
        discrimination_A_fake = outputs[7]
        discrimination_B_real = outputs[8]
        discrimination_B_fake = outputs[9]
        discrimination_A_dot_real = outputs[10]
        discrimination_A_dot_fake = outputs[11]
        discrimination_B_dot_real = outputs[12]
        discrimination_B_dot_fake = outputs[13]

        # Cycle loss.
        cycle_loss = l1_loss(inputs[0], cycle_A) + l1_loss(inputs[1], cycle_B)

        # Identity loss.
        identity_loss = l1_loss(inputs[0], identity_A) + l1_loss(inputs[1], identity_B)

        # Generator loss.
        generator_loss_A2B = l2_loss(tf.ones_like(discrimination_B_fake), discrimination_B_fake)
        generator_loss_B2A = l2_loss(tf.ones_like(discrimination_A_fake), discrimination_A_fake)

        two_step_generator_loss_A = l2_loss(tf.ones_like(discrimination_A_dot_fake), discrimination_A_dot_fake)
        two_step_generator_loss_B = l2_loss(tf.ones_like(discrimination_B_dot_fake), discrimination_B_dot_fake)

        generator_loss = generator_loss_A2B + generator_loss_B2A + two_step_generator_loss_A + \
                         two_step_generator_loss_B + hp.lambda_cycle * cycle_loss + hp.lambda_identity * identity_loss

        discriminator_loss_A_real = l2_loss(tf.ones_like(discrimination_A_real), discrimination_A_real)
        discriminator_loss_A_fake = l2_loss(tf.zeros_like(discrimination_A_fake), discrimination_A_fake)
        discriminator_loss_A = (discriminator_loss_A_real + discriminator_loss_A_fake) / 2

        discriminator_loss_B_real = l2_loss(tf.ones_like(discrimination_B_real), discrimination_B_real)
        discriminator_loss_B_fake = l2_loss(tf.zeros_like(discrimination_B_fake), discrimination_B_fake)
        discriminator_loss_B = (discriminator_loss_B_real + discriminator_loss_B_fake) / 2

        discriminator_loss_A_dot_real = l2_loss(tf.ones_like(discrimination_A_dot_real), discrimination_A_dot_real)
        discriminator_loss_A_dot_fake = l2_loss(tf.zeros_like(discrimination_A_dot_fake), discrimination_A_dot_fake)
        discriminator_loss_A_dot = (discriminator_loss_A_dot_real + discriminator_loss_A_dot_fake) / 2

        discriminator_loss_B_dot_real = l2_loss(tf.ones_like(discrimination_B_dot_real), discrimination_B_dot_real)
        discriminator_loss_B_dot_fake = l2_loss(tf.zeros_like(discrimination_B_dot_fake), discrimination_B_dot_fake)
        discriminator_loss_B_dot = (discriminator_loss_B_dot_real + discriminator_loss_B_dot_fake) / 2

        discriminator_loss = discriminator_loss_A + discriminator_loss_B + discriminator_loss_A_dot + \
                             discriminator_loss_B_dot

    generator_vars = model.generatorA2B.trainable_variables + model.generatorB2A.trainable_variables
    discriminator_vars = model.discriminator_A.trainable_variables + model.discriminator_B.trainable_variables + \
                         model.discriminator_A_dot.trainable_variables + model.discriminator_B_dot.trainable_variables

    grad_gen = gen_tape.gradient(generator_loss, sources=generator_vars)
    grad_dis = dis_tape.gradient(discriminator_loss, sources=discriminator_vars)
    generator_optimizer.apply_gradients(zip(grad_gen, generator_vars))
    discriminator_optimizer.apply_gradients(zip(grad_dis, discriminator_vars))

    gen_loss(generator_loss)
    disc_loss(discriminator_loss)


def sample_train_data(dataset_A, dataset_B):
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= hp.n_frames
        start_A = np.random.randint(frames_A_total - hp.n_frames + 1)
        end_A = start_A + hp.n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= hp.n_frames
        start_B = np.random.randint(frames_B_total - hp.n_frames + 1)
        end_B = start_B + hp.n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B


@tf.function
def test(filename):
    wav, _ = librosa.load(filename, sr=hp.rate)
    f0, timeaxis, sp, ap = world_decompose(wav, hp.rate)
    f0_converted = pitch_conversion(f0, log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B)
    coded_sp = world_encode_spectral_envelop(sp, hp.rate, hp.num_mceps)
    coded_sp_transposed = coded_sp.T
    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
    coded_sp_norm = seg_and_pad(coded_sp_norm, hp.n_frames)

    wav_forms = []
    for i, sp_norm in enumerate(coded_sp_norm):
        sp_norm = np.expand_dims(sp_norm, axis=-1)
        coded_sp_converted_norm = model([sp_norm, sp_norm], training=False)[1][0]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
        coded_sp_converted = np.array(coded_sp_converted, dtype=np.float64).T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decode_sp_converted = world_decode_spectral_envelop(coded_sp_converted, hp.rate)
        if len(f0) < (i + 1) * hp.output_size:
            decode_sp_converted = decode_sp_converted[:len(f0) % hp.output_size]
            f0_piece = f0_converted[i * hp.output_size:i * hp.output_size + len(f0) % hp.output_size]
            ap_piece = ap[i * hp.output_size:i * hp.output_size + len(f0) % hp.output_size]
            wav_transformed = world_speech_synthesis(f0_piece, decode_sp_converted, ap_piece, hp.rate, hp.duration)
            wav_forms.append(wav_transformed)
            break
        else:
            f0_piece = f0_converted[i * hp.output_size:(i + 1) * hp.output_size]
            ap_piece = ap[i * hp.output_size:(i + 1) * hp.output_size]

        wav_transformed = world_speech_synthesis(f0_piece, decode_sp_converted, ap_piece, hp.rate, hp.duration)
        wav_forms.append(wav_transformed)

    wav_forms = tf.convert_to_tensor(np.concatenate(wav_forms), dtype=tf.float64)

    return wav_forms


def seg_and_pad(src, n_frames):
    n_origin = src.shape[1]
    n_padded = (n_origin // n_frames + 1) * n_frames
    left_pad = (n_padded - n_origin) // 2
    right_pad = n_padded - n_origin - left_pad
    src = np.pad(src, [(0, 0), (left_pad, right_pad)], 'constant', constant_values=0)
    src = np.reshape(src, [-1, hp.num_mceps, n_frames])

    return src


if __name__ == '__main__':
    # Training.
    model = CycleGAN2()

    discriminator_optimizer = tf.optimizers.Adam(learning_rate=hp.discriminator_lr)
    generator_optimizer = tf.optimizers.Adam(learning_rate=hp.generator_lr)
    gen_loss = tf.keras.metrics.Mean()
    disc_loss = tf.keras.metrics.Mean()

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.join(hp.logdir, current_time)
    summary_writer = tf.summary.create_file_writer(logdir)

    print('Loading cached data...')
    with open('./datasets/JSUT/jsut.p', 'rb') as f:
        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = pickle.load(f)

    with open('./datasets/target_voice/target_voice.p', 'rb') as f:
        coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = pickle.load(f)

    iteration = 1
    epoch = 0
    while iteration <= hp.num_iterations:
        dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm)
        n_samples = dataset_A.shape[0]

        for i in range(n_samples // hp.batch_size):
            start = i * hp.batch_size
            end = (i + 1) * hp.batch_size
            train_step([dataset_A[start:end], dataset_B[start:end]])

            if iteration > 10000:
                hp.lambda_identity = 0

            if iteration % 2500 == 0:
                model.save_weights(os.path.join(hp.weights_dir, 'weights_{:}'.format(iteration)))

            iteration += 1

        file = np.random.choice(glob.glob('./datasets/JSUT/*.wav'), 1)
        val_wav = tf.expand_dims(test(file), axis=0)

        with summary_writer.as_default():
            tf.summary.scalar('Generator loss', gen_loss.result(), step=epoch)
            tf.summary.scalar('Discriminator loss', disc_loss.result(), step=epoch)
            tf.summary.audio(f'epoch_{epoch}_{file}', val_wav, sample_rate=hp.rate, step=epoch)

        print('Epoch: {} \tGenerator loss: {} \tDiscriminator loss: {}'.format(epoch, gen_loss.result().numpy(),
                                                                               disc_loss.result().numpy()))
        gen_loss.reset_states()
        disc_loss.reset_states()
        epoch += 1

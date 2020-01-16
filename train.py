import tensorflow as tf
import numpy as np
import pickle
import os
import hyperparameter as hp
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


if __name__ == '__main__':
    # Training.
    model = CycleGAN2()

    discriminator_optimizer = tf.optimizers.Adam(learning_rate=hp.discriminator_lr)
    generator_optimizer = tf.optimizers.Adam(learning_rate=hp.generator_lr)
    gen_loss = tf.keras.metrics.Mean()
    disc_loss = tf.keras.metrics.Mean()

    print('Loading cached data...')
    with open('./datasets/my_voice/my_voice.p', 'rb') as f:
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
            else:
                hp.lambda_identity = 10

            if iteration % 2500 == 0:
                model.save_weights(os.path.join(hp.logdir, 'weights_{:}'.format(iteration)))

            iteration += 1

        epoch += 1
        print('Epoch: {} \tGenerator loss: {} \tDiscriminator loss: {}'.format(epoch, gen_loss.result().numpy(),
                                                                               disc_loss.result().numpy()))

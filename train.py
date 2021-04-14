#!/usr/bin/env python3

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import imageio
import numpy as np
import tensorflow as tf
from tensorflow import keras

from net import BPN
from utils import dataset
import utils.utils as ut
import utils.tf_utils as tfu
import utils.np_utils as npu

parser = argparse.ArgumentParser()
parser.add_argument('--color', action='store_true')
opts = parser.parse_args()

TLIST = 'data/train.txt'
VLIST = 'data/val.txt'

if opts.color:
    BSZ = 8
    MAXITER = 19e5
    boundaries = [16e5, 17e5]
else:
    BSZ = 24
    MAXITER = 7e5
    boundaries = [5e5, 5.6e5]

IMSZ = 128
LR = 1e-4
MAXITER = 7e5
VALFREQ = 2e3
SAVEFREQ = 1e5
WTS = 'wts/color' if opts.color else 'wts/grayscale'
if not os.path.exists(WTS):
    os.makedirs(WTS)
log_writer = ut.LogWriter(WTS + '/train.log')

# distributed training strategy
strategy = tf.distribute.MirroredStrategy()
ngpus = strategy.num_replicas_in_sync
GLOBAL_BSZ = BSZ * ngpus
log_writer.log("Using %d GPUs." % ngpus)

# learning rate schedule
values = [LR, np.float32(LR / np.sqrt(10.)), LR / 10.]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

with strategy.scope():
    model = BPN(color=opts.color).model
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    iterations = optimizer.iterations

    if os.path.isfile(WTS+'/opt.npz'):
        ut.loadopt(WTS+'/opt.npz', optimizer, model)
        log_writer.log("Restored optimizer.")
    if os.path.isfile(WTS+'/model.npz'):
        ut.loadmodel(WTS+'/model.npz', model)
        log_writer.log("Restored model.")
    else:
        log_writer.log("No previous checkpoints, new training.")

log_writer.log("Creating dataset.")
train_set, val_set = dataset.create_dataset(
    iterations.numpy(), TLIST, VLIST, bsz=GLOBAL_BSZ, repeats=1,
    patches_per_img=1, height=IMSZ, width=IMSZ, grayscale=(not opts.color))
train_dist_set = strategy.experimental_distribute_dataset(train_set)
val_dist_set = strategy.experimental_distribute_dataset(val_set)


def _one_step(inputs, training=True):
    clean, white_level = tfu.degamma_and_scale(inputs)
    noisy, sig_read, sig_shot = tfu.add_read_shot_noise(clean)
    noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
    net_input = tf.concat([noisy, noise_std], axis=-1)

    if training:
        with tf.GradientTape() as tape:
            basis, coeffs = model(net_input)
            if opts.color:
                denoise, framewise = tfu.apply_filtering_color(
                    noisy, basis, coeffs, True)
                clean = tfu.restore_and_gamma(clean[...,:3], white_level)
            else:
                denoise, framewise = tfu.apply_filtering_gray(
                    noisy, basis, coeffs, True)
                clean = tfu.restore_and_gamma(clean[...,:1], white_level)

            # Loss
            denoise = tfu.restore_and_gamma(denoise, white_level)
            framewise = tfu.restore_and_gamma(framewise, white_level[..., None])
            l2_loss = tfu.l2_loss(denoise, clean) / ngpus
            gradient_loss = tfu.gradient_loss(denoise, clean) / ngpus
            frame_loss, anneal = tfu.frame_loss(framewise, clean, iterations)
            frame_loss = frame_loss / ngpus
            anneal = anneal / ngpus
            psnr = tfu.get_psnr(denoise, clean) / ngpus

            loss = l2_loss + gradient_loss + frame_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    else:
        basis, coeffs = model(net_input)
        if opts.color:
            denoise, framewise = tfu.apply_filtering_color(
                noisy, basis, coeffs, True)
            clean = tfu.restore_and_gamma(clean[...,:3], white_level)
        else:
            denoise, framewise = tfu.apply_filtering_gray(
                noisy, basis, coeffs, True)
            clean = tfu.restore_and_gamma(clean[...,:1], white_level)

        # Loss
        denoise = tfu.restore_and_gamma(denoise, white_level)
        framewise = tfu.restore_and_gamma(framewise, white_level[..., None])
        l2_loss = tfu.l2_loss(denoise, clean) / ngpus
        gradient_loss = tfu.gradient_loss(denoise, clean) / ngpus
        frame_loss, anneal = tfu.frame_loss(framewise, clean, iterations)
        frame_loss = frame_loss / ngpus
        anneal = anneal / ngpus
        psnr = tfu.get_psnr(denoise, clean) / ngpus

        loss = l2_loss + gradient_loss + frame_loss

    lvals = {
        'loss': loss,
        'pixel_l2': l2_loss,
        'gradient_l1': gradient_loss,
        'psnr': psnr,
        'frame_loss': frame_loss,
        'anneal': anneal
    }

    return lvals


@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(
    _one_step, args=(dataset_inputs, True))
  return tfu.custom_replica_reduce(
    strategy, tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_val_step(dataset_inputs):
  per_replica_losses = strategy.run(
    _one_step, args=(dataset_inputs, False))
  return tfu.custom_replica_reduce(
    strategy, tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


stop = ut.getstop()
log_writer.log("Start training.")
for batch in train_dist_set:
    out = distributed_train_step(batch)
    if iterations.numpy() % 100 == 0:
        out = {k + '.t': v for k, v in out.items()}
        out['lr'] = optimizer._decayed_lr('float32').numpy()
        log_writer.log(out, iterations.numpy())
    else:
        log_writer.log({'loss.t': out['loss']}, iterations.numpy())

    ## Validate model every so often
    if iterations.numpy() % VALFREQ == 0 and iterations.numpy() != 0:
        log_writer.log("Validating model")
        for batch in val_dist_set:
            out = distributed_val_step(batch)
            out = {k + '.v': v for k, v in out.items()}
            log_writer.log(out, iterations.numpy())

    if stop[0] or iterations.numpy() >= MAXITER:
        break

# Save model and optimizer state.
if iterations.numpy() > 0:
    log_writer.log("Saving model and optimizer.")
    ut.saveopt(WTS+'/opt.npz', optimizer)
    ut.savemodel(WTS+'/model.npz', model)
log_writer.log("Stopping!")













#!/usr/bin/env python3

import utils.tf_utils as tfu
from net import BPN
import tensorflow as tf
import numpy as np
import imageio
import time
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


psz = 128  # Size of tiles
stride = 64  # Stride of tiles

def group_patches(patches, h, w, psz, stride):
    with tf.GradientTape(persistent=True) as tape:
        ones = tf.ones([1, h, w, tf.shape(patches)[-1]])
        tape.watch(ones)
        ones_patches = tf.image.extract_patches(
            ones, [1, psz, psz, 1], [1, stride, stride, 1],
            [1, 1, 1, 1], 'VALID')
        patches = tf.reshape(patches, ones_patches.shape)

    norm = tape.gradient(ones_patches, ones)
    return tape.gradient(ones_patches, ones, output_gradients=patches) / norm


parser = argparse.ArgumentParser()
parser.add_argument('--color', action='store_true')
parser.add_argument(
    '--gain', default=1, type=int, choices=[1, 2, 4, 8], help='noise level')
opts = parser.parse_args()

if opts.color:
    data = np.load('data/color_testset/%s.npz'%opts.gain)
    noisy_bursts = data['noisy']
    cleans = data['truth']
    white_levels = data['white_level']
    sig_reads = data['sig_read']
    sig_shots = data['sqrt_sig_shot']

    model_path = 'wts/color/model.hdf5'
    bsz = 3
else:
    data = np.load(
        'data/synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz')
    split = {1: 2, 2: 3, 4: 4, 8: 5}[opts.gain]
    noisy_bursts = data['noisy'][73 * split:73 * split + 73].astype(np.float32)
    cleans = data['truth'][73 * split:73 * split + 73].astype(np.float32)
    white_levels = np.ones([73])
    sig_reads = data['sig_read'][73 * split:73 * split + 73].astype(np.float32)
    sig_shots = data['sig_shot'][73 * split:73 * split + 73].astype(np.float32)

    model_path = 'wts/grayscale/model.hdf5'
    bsz = 6

model = BPN(color=opts.color).model
model.load_weights(model_path)
print("Model restored from " + model_path)

psnrs = []

for k in range(sig_reads.shape[0]):

    clean = cleans[k]
    noisy = noisy_bursts[k]
    h, w = noisy.shape[0:2]

    sig_read, sig_shot, white_level = sig_reads[k], sig_shots[k], white_levels[k]

    start_time = time.time()
    h_pad = np.ceil((h - psz) / stride) * stride - (h - psz)
    w_pad = np.ceil((w - psz) / stride) * stride - (w - psz)
    h_pad, w_pad = np.int32(h_pad), np.int32(w_pad)

    if opts.color:
        noisy = tf.reshape(noisy, [h, w, -1])
    noisy = tf.pad(noisy, [[0, h_pad], [0, w_pad], [0, 0]])

    noisy_patches = tf.image.extract_patches(
        noisy[None], [1, psz, psz, 1], 
        [1, stride, stride, 1], [1, 1, 1, 1], 'VALID')
    noisy_patches = tf.reshape(noisy_patches, [-1, psz, psz, noisy.shape[-1]])

    denoise_patches = []
    num_patches = tf.shape(noisy_patches)[0]
    for i in range(num_patches // bsz):
        noisy = noisy_patches[i * bsz:i * bsz + bsz]
        noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
        net_input = tf.concat([noisy, noise_std], axis=-1)

        basis, coeffs = model(net_input)
        if opts.color:
            denoise = tfu.apply_filtering_color(noisy, basis, coeffs)
        else:
            denoise = tfu.apply_filtering_gray(noisy, basis, coeffs)

        denoise_patches.append(denoise)

    denoise = tf.concat(denoise_patches, axis=0)
    denoise = group_patches(denoise, h + h_pad, w + w_pad, psz, stride)
    denoise = denoise[:, :h, :w]

    noisy = tfu.restore_and_gamma(noisy, white_level).numpy()
    clean = tfu.restore_and_gamma(
        clean[..., None], white_level).numpy().squeeze()
    denoise = tfu.restore_and_gamma(denoise, white_level).numpy().squeeze()

    lbuff = 8  # crop this out when reporting psnr, following Mildenhall et al.
    clean = np.clip(clean, 0., 1.)[lbuff:-lbuff, lbuff:-lbuff]
    denoise = np.clip(denoise, 0., 1.)[lbuff:-lbuff, lbuff:-lbuff]

    mse = np.mean(np.square(denoise - clean))
    psnr = np.mean(-10. * np.log10(mse))
    psnrs.append(psnr)

print('Average PSNR: %.2f' % np.mean(psnrs))


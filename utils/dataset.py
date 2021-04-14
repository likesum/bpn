import glob

import numpy as np
import tensorflow as tf


MAX_ITER = 2e6


@tf.function
def gen_burst(
        patches, burst_length, bsz, repeats, height, width,
        upscale, jitter, small_jitter):
    """return a batch of clean bursts, [bsz, h, w, burst_length*channels]"""
    # patches has shape [bsz, h_up, w_up, channels]
    j_up = jitter * upscale
    h_up = height * upscale
    w_up = width * upscale
    channels = tf.shape(patches)[-1]

    bigj_patches = patches
    delta_up = (jitter - small_jitter) * upscale
    smallj_patches = patches[:, delta_up:-delta_up, delta_up:-delta_up, ...]

    unique = bsz // repeats
    batch = []
    for i in range(unique):
        for j in range(repeats):
            curr = [patches[i, j_up:-j_up, j_up:-j_up, :]]
            prob = tf.minimum(tf.cast(tf.random.poisson(
                [], 1.5), tf.float32) / burst_length, 1.)
            for k in range(burst_length - 1):
                flip = tf.random.uniform([])
                p2use = tf.cond(
                    flip < prob, lambda: bigj_patches, lambda: smallj_patches)
                curr.append(
                    tf.image.random_crop(p2use[i, ...], [h_up, w_up, channels]))
            curr = tf.stack(curr, axis=0)
            curr = tf.image.resize(
                curr, [height, width], method=tf.image.ResizeMethod.AREA)
            curr = tf.transpose(curr, [1, 2, 0, 3])
            curr = tf.reshape(curr, [height, width, burst_length * channels])
            batch.append(curr)
    batch = tf.stack(batch, axis=0)
    return batch


@tf.function
def load_patches(
        image, height, width, num_patches, upscale, jitter, is_val=False):
    """Randomly take patches from the image"""
    j_up = jitter * upscale
    h_up = height * upscale + 2 * j_up
    w_up = width * upscale + 2 * j_up
    v_error = tf.maximum((h_up - tf.shape(image)[0] + 1) // 2, 0)
    h_error = tf.maximum((w_up - tf.shape(image)[1] + 1) // 2, 0)
    image = tf.pad(image, [[v_error, v_error], [h_error, h_error], [0, 0]])

    if not is_val:
        stack = []
        for i in range(num_patches):
            stack.append(
                tf.image.random_crop(image, [h_up, w_up, tf.shape(image)[-1]]))
        stack = tf.stack(stack, axis=0)
    else:
        stack = tf.image.extract_patches(
            image[None], [1, h_up, w_up, 1], [1, h_up // 2, w_up // 2, 1],
            [1, 1, 1, 1], 'VALID')
        stack = tf.reshape(stack, [-1, h_up, w_up, tf.shape(image)[-1]])
    return stack


@tf.function
def load_image(filename, channels):
    image = tf.io.read_file(filename)
    if channels == 1:
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32) / 255.
        image = tf.reduce_mean(image, axis=-1, keepdims=True)
    else:
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.cast(image, tf.float32) / 255.
    return image


def create_dataset(
        niter, train_list, val_list, bsz=128, repeats=1, height=128, width=128,
        patches_per_img=4, burst_length=8, upscale=4, jitter=16, 
        small_jitter=2, grayscale=True, **kwargs):

    if grayscale:
        def load_image_fn(filename): return load_image(filename, channels=1)
    else:
        def load_image_fn(filename): return load_image(filename, channels=3)

    def train_load_patches_fn(img): return load_patches(
        img, height, width, patches_per_img, upscale, jitter, False)

    def val_load_patches_fn(img): return load_patches(
        img, height, width, patches_per_img, upscale, jitter, True)

    def gen_burst_fn(patches): return gen_burst(
        patches, burst_length, bsz, repeats,
        height, width, upscale, jitter, small_jitter)

    # Resume training list
    train_files = [l.strip() for l in open(train_list)]
    train_files = train_files * int(np.ceil(MAX_ITER * bsz / len(train_files)))
    np.random.RandomState(123).shuffle(train_files)
    train_files = train_files[niter:]

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_set = tf.data.Dataset.from_tensor_slices(train_files)
    train_set = (train_set
                 .map(load_image_fn, num_parallel_calls=AUTOTUNE)
                 .map(train_load_patches_fn, num_parallel_calls=AUTOTUNE)
                 .unbatch()
                 .batch(bsz // repeats)
                 .map(gen_burst_fn, num_parallel_calls=AUTOTUNE)
                 .prefetch(4))

    val_files = [l.strip() for l in open(val_list)]
    val_set = tf.data.Dataset.from_tensor_slices(val_files)
    val_set = (val_set
               .map(load_image_fn, num_parallel_calls=AUTOTUNE)
               .map(val_load_patches_fn, num_parallel_calls=AUTOTUNE)
               .unbatch()
               .batch(bsz // repeats, drop_remainder=True)
               .map(gen_burst_fn, num_parallel_calls=AUTOTUNE)
               .prefetch(4))

    return train_set, val_set

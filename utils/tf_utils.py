import numpy as np
import tensorflow as tf
import time


def add_read_shot_noise(
    imgs, sig_read=None, sig_shot=None, 
    min_read=-3., max_read=-1.5, min_shot=-2., max_shot=-1.):
    bsz = tf.shape(imgs)[0]
    if sig_read is None or sig_shot is None:
        sig_read = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_read, max_read))
        sig_shot = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_shot, max_shot))
    read = sig_read * tf.random.normal(tf.shape(imgs))
    shot = tf.sqrt(imgs) * sig_shot * tf.random.normal(tf.shape(imgs))
    noisy = imgs + shot + read
    return noisy, sig_read, sig_shot


def estimate_std(noisy, sig_read, sig_shot):
    return tf.sqrt(sig_read**2 + tf.maximum(0., noisy) * sig_shot**2)


def degamma_and_scale(
    imgs, gamma=1. / 2.2, min_scale=0.1, max_scale=1., white_level=None):
    bsz = tf.shape(imgs)[0]
    if white_level is None:
        white_level = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], np.log10(min_scale), np.log10(max_scale)))
    imgs = white_level * (imgs**(1. / gamma))
    return imgs, white_level


def restore_and_gamma(imgs, white_level):
    imgs = imgs / white_level
    return sRGB_transfer(imgs)


def sRGB_transfer(x):
    ''' 
    From https://github.com/google/burst-denoising/blob/master/demosaic_utils.py, 
    which is slightly different from what is described in the paper 
    '''
    b = .0031308
    gamma = 1. / 2.4
    # a = .055
    # k0 = 12.92
    a = 1. / (1. / (b**gamma * (1. - gamma)) - 1.)
    k0 = (1 + a) * gamma * b**(gamma - 1.)

    def gammafn(x): return (1 + a) * tf.pow(tf.maximum(x, b), gamma) - a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0 * x, gammafn(x))
    k1 = (1 + a) * gamma
    srgb = tf.where(x > 1, k1 * x - k1 + 1, srgb)
    return srgb


def get_gradient(imgs):
    return tf.concat([
        .5 * (imgs[:, 1:, :-1, :] - imgs[:, :-1, :-1, :]),
        .5 * (imgs[:, :-1, 1:, :] - imgs[:, :-1, :-1, :])], axis=-1)


def gradient_loss(pred, gt):
    return l1_loss(get_gradient(pred), get_gradient(gt))


def l1_loss(pred, gt):
    return tf.reduce_mean(tf.abs(pred - gt))


def l2_loss(pred, gt):
    return tf.reduce_mean(tf.square(pred - gt))


def frame_loss(framewise, gt, global_step, alpha=.9998, beta=100.):
    burst_length = tf.cast(tf.shape(framewise)[-2], tf.float32)
    gt = tf.broadcast_to(gt[..., None, :], tf.shape(framewise))
    loss = l2_loss(framewise, gt) + gradient_loss(framewise, gt)
    annealing = beta * tf.pow(alpha, tf.cast(global_step, tf.float32))
    return loss * burst_length * annealing, annealing


def get_psnr(pred, gt):
    pred = tf.clip_by_value(pred, 0., 1.)
    gt = tf.clip_by_value(gt, 0., 1.)
    mse = tf.reduce_mean((pred - gt)**2.0, axis=[1, 2, 3])
    psnr = tf.reduce_mean(-10. * tf.math.log(mse) / tf.math.log(10.))
    return psnr


def apply_filtering_gray(imgs, basis, coeffs, intermedia=False):
    """
    Apply per-pixel filtering to input images, same kernels for all channels
    """
    b, h, w, burst_length = imgs.shape

    coeffs = tf.reshape(coeffs, [b, h * w, -1])
    kernels = tf.matmul(coeffs, basis)
    kernels = tf.reshape(kernels, [b, h, w, -1, burst_length])

    ksz = int(np.sqrt(kernels.shape[-2]))
    padding = (ksz - 1) // 2

    imgs = tf.pad(imgs, [[0, 0], [padding, padding], [
                  padding, padding], [0, 0]], 'REFLECT')
    patches = tf.image.extract_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length])
    out = tf.reduce_sum(framewise, axis=-1, keepdims=True)

    if intermedia:
        return out, framewise[..., None] * burst_length
    else:
        return out


def apply_filtering_color(imgs, basis, coeffs, intermedia=False):
    """
    Apply per-pixel filtering to input images, separate kernels for different 
    color channels.
    """
    b, h, w, c = imgs.shape
    burst_length = c // 3

    coeffs = tf.reshape(coeffs, [b, h * w, -1])
    kernels = tf.matmul(coeffs, basis)
    kernels = tf.reshape(kernels, [b, h, w, -1, burst_length * 3])

    ksz = int(np.sqrt(kernels.shape[-2]))
    padding = (ksz - 1) // 2

    imgs = tf.pad(imgs, [[0, 0], [padding, padding], [
                  padding, padding], [0, 0]], 'REFLECT')
    patches = tf.image.extract_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length * 3])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length, 3])
    out = tf.reduce_sum(framewise, axis=-2)

    if intermedia:
        return out, framewise * burst_length
    else:
        return out


def custom_replica_reduce(strategy, reduce_op, tensor_dict, axis):
    return {k: strategy.reduce(
        reduce_op, v, axis) for k, v in tensor_dict.items()}





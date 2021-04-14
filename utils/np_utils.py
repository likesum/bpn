import numpy as np
import imageio

def imsave(nm, img):
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    img = np.uint8(np.clip(img,0,1) * 255.)
    imageio.imsave(nm, img)


def margin_concat(imgs, margin=5, horizontal=True):
    h, w, c = imgs[0].shape
    if horizontal:
        outputs = [np.concatenate([np.ones((h,margin,c)), img], axis=1) for img in imgs[1:]]
        outputs = [imgs[0]] + outputs
        return np.concatenate(outputs, axis=1)
    else:
        outputs = [np.concatenate([np.ones((margin,w,c)), img], axis=0) for img in imgs[1:]]
        outputs = [imgs[0]] + outputs
        return np.concatenate(outputs, axis=0)


def save_burst(nm, burst):
    ''' Expect an input of shape [H, W, BURST_LENGTH] '''
    imgs = np.split(burst, burst.shape[-1], axis=-1)

    if nm.endswith('.gif'):
        imgs = [np.uint8(np.clip(img,0,1) * 255.) for img in imgs]
        imageio.mimsave(nm, imgs, duration=0.5)
    else:
        output = margin_concat(imgs)
        imsave(nm, output)


def save_color_burst(nm, burst):
    ''' Expect an input of shape [H, W, C, BURST_LENGTH] '''
    imgs = np.split(burst, burst.shape[-1], axis=-1)
    imgs = [np.squeeze(img) for img in imgs]

    if nm.endswith('.gif'):
        imgs = [np.uint8(np.clip(img,0,1) * 255.) for img in imgs]
        imageio.mimsave(nm, imgs, duration=0.5)
    else:
        output = margin_concat(imgs)
        imsave(nm, output)


def get_mse(pred, gt):
    return np.mean(np.square(pred-gt))

def get_psnr(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = np.mean((pred-gt)**2.0)
    psnr = np.mean(-10.*np.log10(mse))
    return psnr

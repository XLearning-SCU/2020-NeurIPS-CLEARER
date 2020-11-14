import os
import numpy as np
import math
import torch
import random
import shutil
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.nn import init

class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

# randomly crop a patch from image
def crop_patch(im, pch_size):
  H = im.shape[0]
  W = im.shape[1]
  ind_H = random.randint(0, H-pch_size)
  ind_W = random.randint(0, W-pch_size)
  pch = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size]
  return pch

# crop an image to the multiple of base
def crop_img(image, base=64):
  h = image.shape[0]
  w = image.shape[1]
  crop_h = h % base
  crop_w = w % base
  return image[crop_h//2:h-crop_h+crop_h//2, crop_w//2:w-crop_w+crop_w//2, :]

# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
  assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
  H = image.shape[0]
  W = image.shape[1]
  patches = []
  image_padding = np.pad(image,((overlap,overlap),(overlap,overlap),(0,0)),mode='edge')
  for h in range(H//patch_size):
    for w in range(W//patch_size):
      idx_h = [h*patch_size,(h+1)*patch_size+overlap]
      idx_w = [w*patch_size,(w+1)*patch_size+overlap]
      patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1],idx_w[0]:idx_w[1],:],axis=0))
  return np.concatenate(patches,axis=0)

# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
  assert len(image_size) > 1
  assert patches.shape[-3] == patches.shape[-2]
  H = image_size[0]
  W = image_size[1]
  patch_size = patches.shape[-2]-overlap
  image = np.zeros(image_size)
  idx = 0
  for h in range(H//patch_size):
    for w in range(W//patch_size):
      image[h*patch_size:(h+1)*patch_size,w*patch_size:(w+1)*patch_size,:] = patches[idx,overlap:patch_size+overlap,overlap:patch_size+overlap,:]
      idx += 1
  return image

def inverse_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image,k=-1)
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, k=-1)
    elif mode == 4:
        out = np.rot90(image, k=-2)
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=-2)
    elif mode == 6:
        out = np.rot90(image, k=-3)
    elif mode == 7:
        out = np.flipud(image)
        out = np.rot90(out, k=-3)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def save(model, model_path):
  torch.save({'state_dict':model.state_dict(), 'arch_param': model.arch_parameters()[0]}, model_path)

# pred and tar (B, C, H, W)
def compute_psnr(pred, tar):
  assert pred.shape == tar.shape
  pred = pred.transpose(0,2,3,1)
  tar = tar.transpose(0,2,3,1)
  psnr = 0
  for i in range(pred.shape[0]):
    psnr += compare_psnr(tar[i],pred[i])
  return psnr/pred.shape[0]
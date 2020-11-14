import os
import sys
import numpy as np
import torch
import time
import logging
import glob
import argparse
import torchvision
from PIL import Image

import utils
import pytorch_ssim

parser = argparse.ArgumentParser("deraining")
parser.add_argument('--data', type=str, default='datasets/rain800/test_syn/*.jpg', help='location of the data corpus')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--model_path', type=str, default='eval-EXP-20201112-191755/last_weights.pt')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled=True

    model = torch.load(args.model_path)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    psnr, ssim = infer(args.data, model)
    logging.info('psnr:%6f ssim:%6f', psnr, ssim)

def infer(data_path, model):
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()

    model.eval()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    with torch.no_grad():
        for step, pt in enumerate(glob.glob(data_path)):
            image = np.array(Image.open(pt))

            clear_image = utils.crop_img(image[:,:image.shape[1]//2,:], base=args.patch_size)
            rain_image = utils.crop_img(image[:,image.shape[1]//2:,:], base=args.patch_size)

            # # Test on whole image
            # input = transforms(rain_image).unsqueeze(dim=0).cuda()
            # target = transforms(clear_image).unsqueeze(dim=0).cuda(async=True)
            # logits = model(input)
            # n = input.size(0)

            # Test on whole image with data augmentation
            target = transforms(clear_image).unsqueeze(dim=0).cuda()
            for i in range(8):
                im = utils.data_augmentation(rain_image,i)
                input = transforms(im.copy()).unsqueeze(dim=0).cuda()
                begin_time = time.time()
                if i == 0:
                    logits = utils.inverse_augmentation(model(input).cpu().numpy().transpose(0,2,3,1)[0],i)
                else:
                    logits = logits + utils.inverse_augmentation(model(input).cpu().numpy().transpose(0,2,3,1)[0],i)
                end_time = time.time()
            n = input.size(0)
            logits = transforms(logits/8).unsqueeze(dim=0).cuda()

            # # Test on patches2patches
            # noise_patches = utils.slice_image2patches(rain_image, patch_size=args.patch_size)
            # image_patches = utils.slice_image2patches(clear_image, patch_size=args.patch_size)
            # input = torch.tensor(noise_patches.transpose(0,3,1,2)/255.0, dtype=torch.float32).cuda()
            # target = torch.tensor(image_patches.transpose(0,3,1,2)/255.0, dtype=torch.float32).cuda()
            # logits = model(input)
            # n = input.size(0)

            s = pytorch_ssim.ssim(torch.clamp(logits,0,1), target)
            p = utils.compute_psnr(np.clip(logits.detach().cpu().numpy(),0,1), target.detach().cpu().numpy())
            psnr.update(p, n)
            ssim.update(s, n)
            print('psnr:%6f ssim:%6f' % (p, s))

            # Image.fromarray(rain_image).save(args.save+'/'+str(step)+'_noise.png')
            # Image.fromarray(np.clip(logits[0].cpu().numpy().transpose(1,2,0)*255, 0, 255).astype(np.uint8)).save(args.save+'/'+str(step)+'_denoised.png')

    return psnr.avg, ssim.avg

if __name__ == '__main__':
  main()
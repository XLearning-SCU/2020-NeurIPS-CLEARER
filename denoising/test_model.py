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

parser = argparse.ArgumentParser("denoising")
parser.add_argument('--data', type=str, default='./datasets/BSD500/test/*.jpg', help='location of the data')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--model_path', type=str, default='eval-EXP-20201112-173215/best_loss_weights.pt')
parser.add_argument('--sigma', type=int, default=30, help='noise sigma')
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

    psnr, ssim, time = infer(args.data, model)
    logging.info('psnr:%6f ssim:%6f time:%6f', psnr, ssim, time)

def infer(data_path, model):
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()
    times = utils.AvgrageMeter()

    model.eval()
    rgb2gray = torchvision.transforms.Grayscale(1)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    with torch.no_grad():
        for step, pt in enumerate(glob.glob(data_path)):
            image = utils.crop_img(np.array(rgb2gray(Image.open(pt)))[..., np.newaxis])
            noise_map = np.random.randn(*(image.shape))*args.sigma
            noise_img = np.clip(image+noise_map,0,255).astype(np.uint8)

            # # Test on whole image
            # input = transforms(noise_img).unsqueeze(dim=0).cuda()
            # target = transforms(image).unsqueeze(dim=0).cuda()
            # begin_time = time.time()
            # logits = model(input)
            # end_time = time.time()
            # n = input.size(0)

            # Test on whole image with data augmentation
            target = transforms(image).unsqueeze(dim=0).cuda()
            for i in range(8):
                im = utils.data_augmentation(noise_img,i)
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
            # noise_patches = utils.slice_image2patches(noise_img, patch_size=64)
            # image_patches = utils.slice_image2patches(image, patch_size=64)
            # input = torch.tensor(noise_patches.transpose(0,3,1,2)/255.0, dtype=torch.float32).cuda()
            # target = torch.tensor(image_patches.transpose(0,3,1,2)/255.0, dtype=torch.float32).cuda()
            # begin_time = time.time()
            # logits = model(input)
            # end_time = time.time()
            # n = input.size(0)

            s = pytorch_ssim.ssim(torch.clamp(logits,0,1), target)
            p = utils.compute_psnr(np.clip(logits.detach().cpu().numpy(),0,1), target.detach().cpu().numpy())
            t = end_time-begin_time
            psnr.update(p, n)
            ssim.update(s, n)
            times.update(t,n)
            print('psnr:%6f ssim:%6f time:%6f' % (p, s, t))
            
            # Image.fromarray(noise_img[...,0]).save(args.save+'/'+str(step)+'_noise.png')
            # Image.fromarray(np.clip(logits[0,0].cpu().numpy()*255, 0, 255).astype(np.uint8)).save(args.save+'/'+str(step)+'_denoised.png')

    return psnr.avg, ssim.avg, times.avg

if __name__ == '__main__':
  main()
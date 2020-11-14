import os
import sys
import time
import glob
import math
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn

import utils
import pytorch_ssim
from search_space import HighResolutionNet
from datasets import PatchSimulateDataset

parser = argparse.ArgumentParser("Denoising")
parser.add_argument('--data', type=str, default='./datasets/BSD500/', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--steps', type=int, default=100, help='steps of each epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--gpu', type=str, default='1', help='gpu device ids')
parser.add_argument('--sigma', type=int, default=30, help='noise level')
parser.add_argument('--ckt_path', type=str, default='search-EXP-20201112-143409/last_weights.pt', help='checkpoint path of search')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

MSELoss = torch.nn.MSELoss().cuda()

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  logging.info("args = %s", args)

  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.enabled=True

  model = HighResolutionNet(ckt_path=args.ckt_path)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  train_samples = PatchSimulateDataset(args.data+'train/', args.steps*args.batch_size, args.patch_size, sigma=args.sigma)
  val_samples = PatchSimulateDataset(args.data+'test/', 50*args.batch_size, args.patch_size, sigma=args.sigma)

  train_queue = torch.utils.data.DataLoader(train_samples, batch_size=args.batch_size, pin_memory=True)
  valid_queue = torch.utils.data.DataLoader(val_samples, batch_size=args.batch_size, pin_memory=True)

  best_psnr = 0
  best_psnr_epoch = 0
  best_ssim = 0
  best_ssim_epoch = 0
  best_loss = float("inf") 
  best_loss_epoch = 0
  for epoch in range(args.epochs):
    logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, scheduler.get_lr()[0])
    
    # training
    train(train_queue, model, optimizer)
    # validation
    psnr, ssim, loss = infer(valid_queue, model)
    
    if psnr > best_psnr and not math.isinf(psnr):
      torch.save(model, os.path.join(args.save, 'best_psnr_weights.pt'))
      best_psnr_epoch = epoch+1
      best_psnr = psnr
    if ssim > best_ssim:
      torch.save(model, os.path.join(args.save, 'best_ssim_weights.pt'))
      best_ssim_epoch = epoch+1
      best_ssim = ssim
    if loss < best_loss:
      torch.save(model, os.path.join(args.save, 'best_loss_weights.pt'))
      best_loss_epoch = epoch+1
      best_loss = loss

    scheduler.step()
    logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)

  logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
  torch.save(model, os.path.join(args.save, 'last_weights.pt'))


def train(train_queue, model, optimizer):
  for step, (input, target) in enumerate(train_queue):
    print('--steps:%d--' % step)
    model.train()
    input = input.cuda()
    target = target.cuda()
    optimizer.zero_grad()
    logits = model(input)
    loss = MSELoss(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

def infer(valid_queue, model):
  psnr = utils.AvgrageMeter()
  ssim = utils.AvgrageMeter()
  loss = utils.AvgrageMeter()

  model.eval()
  with torch.no_grad():
    for _, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()
      logits = model(input)
      l = MSELoss(logits, target)
      s = pytorch_ssim.ssim(torch.clamp(logits,0,1), target)
      p = utils.compute_psnr(np.clip(logits.detach().cpu().numpy(),0,1), target.detach().cpu().numpy())
      n = input.size(0)
      psnr.update(p, n)
      ssim.update(s, n)
      loss.update(l, n)
  return psnr.avg, ssim.avg, loss.avg

if __name__ == '__main__':
  main()
#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import argparse
from datetime import datetime
import os, sys
import cv2
import numpy as np
from models import *
import math
import glob
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import scipy.io as sio

from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
from utils.common_utils import *
from torch.distributions import Poisson
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PR_Poisson_noise")
parser.add_argument("--imgs-dir", type=str, default='Unatural256', help="directory of testing images")
parser.add_argument("--masktype", type=str, default='bipolar', help="type of mask")
parser.add_argument("--noisetype", type=str, default='poisson', help="type of noise")
parser.add_argument("--nummask", type=int, default=3, help="number of masks")
parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
parser.add_argument("--numits", type=int, default=10000,help="Number of training iterations")

opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpuid)
torch.set_num_threads(8)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# define the add noise module
def add_noise(model, nlevel):
    for n in [x for x in model.parameters() if len(x.size()) == 4]: 
        noise = torch.randn(n.size())*nlevel
        noise = noise.type(dtype)
        n.data = n.data + noise

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.__stdout__
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() 
#        self.close()
    def flush(self):
        self.log.flush()  
        
        
input_depth = 32
lr = 0.01 # learning rate

INPUT =     'noise'

reg_noise_std = 0.01 # the input noise helps the performance
psnr_trace=[]
psnr_noavg_trace = []
loss_trace=[]
trace ={}

burnin_iter = 2000
weight_decay = 5e-8 # help for the performance

sgld_mean_each = 0
sgld_mean_tmp = 0

class Ax_pr_cdp(nn.Module):
    def __init__(self, c,h,w,masktype):
        super(Ax_pr_cdp, self).__init__()
        self.c=c
        self.h=h
        self.w=w
        if masktype == 'uniform':
            ang = 2*torch.acos(torch.zeros(1)).item()*2*torch.rand(c,h,w,1)
            self.mask = torch.cat((torch.cos(ang),torch.sin(ang)), 3).cuda()
        elif masktype == 'bipolar':            
            self.mask = 2.0*(torch.bernoulli(0.5*torch.ones(c,h,w,2))-0.5).cuda()

    def forward(self, img):
        img_s=img.reshape(1,self.h,self.w,1).repeat(self.c,1,1,2)*self.mask
        meas=torch.fft(img_s,2,normalized=True)
        meas= torch.sqrt(meas[:,:,:,0]**2+meas[:,:,:,1]**2)             
        return meas  
 
if opts.noisetype == 'poisson':
    alpha = 9 # Noise level, choices from [9,27, 81]
    if opts.imgs_dir == 'Natural256':
        nlevel = 1e-10 # tuneable for different noise level, which depends on the noise level and the accuracy of the estimated noise
    else:
        nlevel = 1e-12 # the smaller the alpha value, the smaller the nlevel is
elif opts.noisetype == 'awgn':
    alpha = 10 #[10, 15, 20]
    if opts.imgs_dir == 'Natural256':
        nlevel = 1e-5 # tuneable for different noise level, which depends on the noise level and the accuracy of the estimated noise
    else:
        nlevel = 1e-7 # Generally for unnatural256 images, we set nlevel smaller for better performance
else:
    alpha = 0
    nlevel = 0.0001

file_name = glob.glob('./PRImageSet/'+opts.imgs_dir+'/*.png' )
data_num = len(file_name)    
psnr_a = np.zeros((data_num,)) 
ssim_a = np.zeros((data_num,))


psnr_aver = 0.0
ssim_aver = 0.0
num_iter = opts.numits
for fi,Img_Name in enumerate(file_name): #[0:1]
    MODEL_PATH = './Github_PR_est_noise_poi_FINAL/PR_Recon_Results/%s_alpha%2d/%s/' % (opts.noisetype, opts.masktype, opts.imgs_dir,alpha,Img_Name[Img_Name.rfind('/')+1:-4])
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    sgld_mean_each = 0
    sgld_mean_tmp = 0
    img = np.array(cv2.imread(Img_Name, -1), dtype=np.float32)/255.
    if img.ndim == 2:
        Img = np.expand_dims(img, axis=0)
    else:
        Img = img.transpose(2,0,1)
    c,w,h = Img.shape
    Ax=Ax_pr_cdp(opts.nummask,w,h,opts.masktype)
    Img = np.expand_dims(Img,axis=1)
    Img_tensor = torch.FloatTensor(Img).cuda()

    meas_nf = Ax(Img_tensor) 
    if opts.noisetype == 'poisson':        
        meas_nf_before = meas_nf
        intensity_noise=alpha/255.0*(meas_nf)*torch.randn(meas_nf.shape).cuda()

        y2=meas_nf**2+intensity_noise
        y2=torch.clamp(y2,min=0.0)
    
        Measure= torch.sqrt(y2)
        error = torch.mean((Measure - meas_nf_before)**2) # oracle noise, which is not accessiable 
        error_est = (alpha/255.0)**2/4 #/(Measure.shape[0]*Measure.shape[1]*Measure.shape[2])
        sys.stdout = Logger(MODEL_PATH+'results.txt')
        print('error {:f} error_est {:f}'.format(error, error_est))
    elif opts.noisetype == 'awgn':  
        noise_std=torch.randn(meas_nf.shape).cuda()
        noise = noise_std*torch.norm(meas_nf)/torch.norm(noise_std)/float(np.sqrt(10.0**(alpha/10.0)))

        y2=meas_nf+noise
        Measure=torch.clamp(y2,min=0.0) 
        error = torch.mean((Measure - meas_nf)**2) 
        error_est = torch.mean(Measure**2)/(float(np.sqrt(10.0**(alpha/10.0))))**2
        sys.stdout = Logger(MODEL_PATH+'results.txt')
        print('error {:f} error_est {:f}'.format(error, error_est))
    else :
        sys.stdout = Logger(MODEL_PATH+'results.txt')
        print('no noise')
        Measure = meas_nf
        error = 0


    img_name = Img_Name
    net_input = get_noise(input_depth, INPUT, (w,h)).type(dtype).detach()

    NET_TYPE = 'skip' #'skip' # UNet, ResNet
    net = get_net(input_depth, NET_TYPE, 'reflection',
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear',
                    n_channels=c).type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)


# # Define closure and optimize

    def closure():
        global i, num_iter, net_input, psnr_trace, psnr_noavg_trace, loss_trace, trace, sgld_mean_tmp, sgld_mean_each
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    
        Img_rec = net(net_input)

        net_output = Ax(Img_rec)

    
        total_loss = mse(net_output, Measure)
        
            
        total_loss.backward()

        if i > burnin_iter:
            sgld_mean_each += Img_rec
            sgld_mean_tmp = sgld_mean_each / (i-burnin_iter)
        else:
            sgld_mean_tmp = Img_rec
    
        if (i + 1) % 10 == 0:
            psnr_noavg = compare_psnr(np.squeeze(torch_to_np(Img_rec)),img,1.)
            ssim_noavg = compare_ssim(np.squeeze(torch_to_np(Img_rec)),img,data_range = 1.)
            psnr = compare_psnr(np.squeeze(torch_to_np(sgld_mean_tmp)),img,1.)
            ssim = compare_ssim(np.squeeze(torch_to_np(sgld_mean_tmp)),img,data_range = 1.)
            now = datetime.now()
            sys.stdout = Logger(MODEL_PATH+'results.txt')
            print(img_name, "loss in ", i + 1, ":", total_loss.item(),"psnr_noavg:",psnr_noavg, "psnr:",psnr, now.strftime("%H:%M:%S"))
            
            loss_trace = np.append(loss_trace,total_loss.item())
            psnr_trace= np.append(psnr_trace,psnr)
            psnr_noavg_trace = np.append(psnr_noavg_trace,psnr_noavg)
            
            trace['loss'] = loss_trace
            
            trace['psnr'] = psnr_trace
            trace['psnr_noavg'] = psnr_noavg_trace
            
            sio.savemat(MODEL_PATH+'trace.mat',trace)

            if i == num_iter - 1:
                # save the image
                plt.plot(psnr_trace)
                plt.plot(psnr_noavg_trace)
                plt.savefig(os.path.join(MODEL_PATH,'comp.png'))
        if (i + 1) % 500 == 0 :
            cv2.imwrite(MODEL_PATH+img_name[img_name.rfind('/')+1:-4] + '_bipolar_poisson_%2d_itn%d_%0.4f_%0.2f.png'%(alpha,i+1,psnr,ssim), np.int32(255*np.squeeze(torch_to_np(sgld_mean_tmp))))                      
    
        i += 1
        
        return total_loss


    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    i = 0

    sys.stdout = Logger(MODEL_PATH+'results.txt')
    print('Starting optimization with ASGLD')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[5000, 6000, 7000], gamma=0.5)  # learning rates
    stop_iter = -1
    for j in range(num_iter):
        scheduler.step(j)
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()
        nlevel_ = nlevel*math.exp(30*((error_est/loss)-1))
        add_noise(net, nlevel_)
    
    
    
    
    
    Img_rec = np.squeeze(torch_to_np(sgld_mean_tmp))
    
    torch.save({
        'net': net,
        'net_input': net_input,
        'iters': i + 1,
        'net_state_dict': net.state_dict(),
    }, MODEL_PATH +img_name[img_name.rfind('/')+1:-4]+ 'checkpoint_best.pth')

    psnr_single = compare_psnr(img.reshape(h,w),Img_rec.reshape(h,w),1.)
    ssim_single = compare_ssim(img.reshape(h,w),Img_rec.reshape(h,w),data_range = 1.) #compare_ssim(Img.reshape(h,w),Img_rec/255.0,data_range = 1.)
    psnr_aver+=psnr_single
    ssim_aver+=ssim_single
    psnr_a[fi]=psnr_single
    ssim_a[fi]=ssim_single        
    sys.stdout = Logger(MODEL_PATH+'psnr.txt')
    print("alpha:", alpha, img_name, "psnr:",psnr_single,ssim_single)

psnr_aver /= len(file_name)
ssim_aver /= len(file_name)
sys.stdout = Logger(MODEL_PATH+'psnr.txt')
print("alpha:", alpha, "average psnr over the image set:", psnr_aver,"average ssim over the image set:", ssim_aver,'\n')    
sio.savemat('./Github_PR_est_noise_poi_FINAL/PR_Recon_Results/%s_PSNR_SSIM_%s_%s_DIP.mat'%(opts.imgs_dir,opts.masktype,opts.noisetype), {'psnr_a':psnr_a,'ssim_a':ssim_a})





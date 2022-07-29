import os
import cv2
import torch.nn.functional as F
fname = 'data/CS/Set11/' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = False
psnr_total=[]
Imgname=sorted(os.listdir(fname))
kk=0
from scipy.linalg import sqrtm
n = 33*33
m = int(n/2.5)
A = torch.empty(n,m).normal_(0, 1)
B=torch.mm(torch.transpose(A,1,0),A)
B=torch.inverse(B)
B=sqrtm(B)
B=torch.Tensor(B)
A=torch.mm(A,B)
A=A.type(dtype)
for imgn in Imgname:
    kk=kk+1
    img_pil = crop_image(get_image(fname+imgn, imsize)[0], d=1)
    img_np=pil_to_np(img_pil)
    img_np = img_np
    img_var=torch.tensor(img_np).type(dtype)
    block_size=33
    c,w,h = img_var.shape
    pad_right = block_size - w%block_size
    pad_bottom = block_size - h%block_size
    padd = (0,pad_bottom,0,pad_right)
    img_var=img_var.unsqueeze(0)
    def forwardm(Img,Phi_input,pad,block_size):
        Img_pad = F.pad(Img, pad, mode='constant', value=0)
 
        p,c,w,h = Img_pad.size()
        Img_col = torch.reshape(Img_pad,(p,c,-1,block_size,h))
        n = Img_col.size()[2]
        Img_col = Img_col.reshape((p,c,n,block_size,-1,block_size))
        Img_col = Img_col.permute(0,1,2,4,3,5)
        Img_col = Img_col.reshape(p,c,-1,block_size*block_size)

        Img_cs = torch.matmul(Img_col, Phi_input)
        return Img_cs
    measurement=forwardm(img_var,A,padd,33)

    reg_noise_std = 1./40.
    LR = 0.002
  
    show_every = 100
    exp_weight=0.999

    num_iter = 20000
    input_depth = 16 
    figsize = 4 
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    net = get_net(input_depth,'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                n_channels=1,
                upsample_mode='bilinear').type(dtype)   
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_torch = np_to_torch(img_np).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None

    last_net = None
    psrn_noisy_last = 0
    i = 0
    burn_in=5000
    psrn_gt_sm=0

    def closure():
        
        global i, out_avg, psrn_noisy_last, last_net, net_input,img_torch,psrn_gt_sm,loss_data
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        #out_2 = out[:,3:6,:,:]
        #out = out[:,0:3,:,:]
        # Smoothing
        if i>burn_in:
            if out_avg is None:
                out_avg=out.detach()
            else:
                #out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                out_avg = out_avg * (i-burn_in-1)/(i-burn_in) + out.detach() * 1/(i-burn_in)
        #noise_disturb= get_noise(3, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        total_loss = mse(forwardm(out,A,padd,33),measurement)
        total_loss.backward()
        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        if i % show_every==0:
            print('psnr:',psrn_gt)
        if i>burn_in:
            psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        if i % show_every==0 and i>burn_in:
            print(imgn)
            print(kk)
            print(i)
            print(psrn_gt_sm)
            print(psnr_total)
            print(sum(psnr_total)/kk)
        i += 1
        
        return total_loss


    optimizer = Adam(0.000001,[{'params':net.parameters()}], lr=LR) # for 40% sampling ratio,
    #optimizer = Adam(0.000002,[{'params':net.parameters()}], lr=LR) for 25% sampling ratio,
    #optimizer = Adam(0.000003,[{'params':net.parameters()}], lr=LR) for 10% sampling ratio,
    for _ in range(num_iter):
        nlevel=1
        optimizer.zero_grad()
        closure()
        optimizer.step(1)
    path='result/Compressed_sensing/no_noise/'
    path=path+imgn
    out_img=torch.clamp(out_avg.squeeze().cpu(),0.,1.)
    out_img=out_img.detach().numpy()
    print(out_img.shape)
    cv2.imwrite(path,255*out_img)
    psnr_total.append(psrn_gt_sm)
    print(psnr_total)
print(sum(psnr_total)/11)
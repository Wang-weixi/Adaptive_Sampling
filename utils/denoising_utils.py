import os
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np

def get_blur_image(img_path):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
        
    """
    imsize=-1
    path=img_path+'/1'
    c='.png'
    d=path+'.png'
    AAA = crop_image(get_image(d, imsize)[0], d=32)
    AAA = pil_to_np(AAA)
    for i in range (2,16):
        b=str(i)
        d=img_path+'/'+b+c
        img_pil = crop_image(get_image(d, imsize)[0], d=32)
        img_np_0 = pil_to_np(img_pil)
        AAA=AAA+img_np_0
    ''' 
    img_pil = crop_image(get_image('data/pair/000/00000000.png', imsize)[0], d=32)
    img_np_0 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000001.png', imsize)[0], d=32)
    img_np_1 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000002.png', imsize)[0], d=32)
    img_np_2 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000003.png', imsize)[0], d=32)
    img_np_3 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000004.png', imsize)[0], d=32)
    img_np_4 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000005.png', imsize)[0], d=32)
    img_np_5 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000006.png', imsize)[0], d=32)
    img_np_6 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000007.png', imsize)[0], d=32)
    img_np_7 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000008.png', imsize)[0], d=32)
    img_np_8 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000009.png', imsize)[0], d=32)
    img_np_9 = pil_to_np(img_pil)
    img_pil = crop_image(get_image('data/pair/000/00000010.png', imsize)[0], d=32)
    img_np_10 = pil_to_np(img_pil)
    '''
    
    #img_blur_np = np.clip((img_np_0+img_np_1+img_np_2+img_np_3+img_np_4+img_np_5+img_np_6+img_np_7+img_np_8+img_np_9+img_np_10)/11, 0, 1).astype(np.float32)
    img_blur_np=np.clip((AAA)/15, 0, 1).astype(np.float32)
    img_blur_pil = np_to_pil(img_blur_np)

    return img_blur_pil, img_blur_np
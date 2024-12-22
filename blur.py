import PIL
from PIL import Image
from matplotlib import cm
import cv2
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

class Blur:
    def __init__(self, img):
        self.png_img = Image.open(img)
        self.img = self.png_img.resize((244, 244))
        self.img = np.array(self.img)
        self.img = torch.tensor(self.img)
        self.img = torch.permute(self.img, (2, 0, 1)).float()
        self.img = self.img.unsqueeze(0)
        
    def kernel_init(self, type, size, sigma):

        if type == 'guassian':
            kernel = torch.zeros((size, size))
            center = size // 2
            
            for i in range(size):
                for j in range(size):
                    x = i - center
                    y = j - center
                    exponent = torch.tensor(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                    kernel[i, j] = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(exponent)
            
            kernel = kernel / torch.sum(kernel)
            return kernel
        elif type == 'default':
            kernel = torch.ones(size, size)
            kernel = kernel / kernel.sum(1, keepdim=True)
            return kernel 


    def blur(self, in_channels, kernel_size):
        kernel = self.kernel_init('default', kernel_size, 1)
        kernel = kernel.unsqueeze(0).unsqueeze(0) 
        kernel = kernel.repeat(in_channels, 1, 1, 1)

        with torch.no_grad():
            if len(self.img.shape) == 3:
                self.img = self.img.unsqueeze(0)
                
            out = F.conv2d(self.img, kernel, groups=in_channels)
            out = out.permute(0, 2, 3, 1)

        blur_img = out.squeeze(0).detach().cpu().numpy() 
        blur_img = np.clip(blur_img, 0, 1)

        print(blur_img.shape)
        cv2.imshow('main', blur_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = 'snowsper.png'
    my_blur = Blur(img)
    my_blur.blur(4, 3)






from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class Filter:
    def __init__(self, img):
        self.img = Image.open(img)
        self.img = np.array(self.img)
        self.img = torch.tensor(self.img).float()
        self.img = self.img.unsqueeze(0)
        self.img = self.img.permute(0, 3, 1, 2)
        
    def kernel_init(self, type, channels, size, sigma):

        if type == 'guassian':
            print("Guassian blur uploading...")
            x = np.arange(size) - size // 2
            y = np.arange(size) - size // 2
            x, y = np.meshgrid(x, y)

            kernel = (1 / (2*np.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
            kernel = torch.tensor(kernel).float()
            kernel = kernel / kernel.sum(1, keepdim=True)
            kernel = kernel.unsqueeze(0)
            kernel = kernel.repeat(4, 1, 1, 1)
            return kernel
        
        elif type == 'default':
            print("Default blur uploading...")
            kernel = torch.ones((channels, size, size)).float()
            kernel = kernel / kernel.sum(1, keepdim=True)
            kernel = kernel.unsqueeze(1)
            return kernel

        elif type == 'vert edge':
            print("Detecting vertical edges...")
            kernel = torch.zeros((size, size)).tolist()

            for i in range(size):
                for j in range(size):
                    if j == (size // 2):
                        kernel[i][j] = 0
                    elif j % 2 == 1:
                        if j > (size // 2):
                            if i > (size//2):
                                kernel[i][j] = kernel[i-1][j] - 1
                            else:
                                kernel[i][j] = i+1
                        else:
                            if i > (size//2):
                                kernel[i][j] = -(abs(kernel[i-1][j]) - 1)
                            else:
                                kernel[i][j] = -(i+1)
                    else:
                        if j > (size // 2):
                            if i > (size // 2):
                                kernel[i][j] = kernel[i-1][j] - 2
                            else:
                                kernel[i][j] = (2 * (i + 1))
                        else:
                            if i > (size // 2):
                                kernel[i][j] = -(abs(kernel[i-1][j]) - 2)
                            else:
                                kernel[i][j] = -(2 * (i + 1))

            kernel = torch.tensor(kernel).unsqueeze(0).float()      
            kernel = kernel.repeat(channels, 1, 1, 1)
            return kernel 
        
    def filter(self, in_channels, kernel_size, type):
        kernel = self.kernel_init(type, in_channels, kernel_size, 2000)

        img = F.conv2d(self.img, kernel, groups=4)
        img = img.permute(0, 2, 3, 1).squeeze(0).squeeze(0)
        img = img.detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        cv2.imshow('main', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = 'snowsper.png'
    my_blur = Filter(img)
    my_blur.filter(4, 5, 'vert edge')

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
        print(self.img.shape)
        self.img = self.img.permute(0, 3, 1, 2)
    
    def guassian_blur(self, channels, size, sigma):
        print("Applying Guassian Blur...")
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        x, y = np.meshgrid(x, y)

        kernel = (1 / (2*np.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
        kernel = torch.tensor(kernel).float()
        kernel = kernel / kernel.sum(1, keepdim=True)
        kernel = kernel.unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.filter(kernel, channels)
    
    def default_blur(self, channels, size):
        print("Applying Default Blur...")
        kernel = torch.ones((channels, size, size)).float()
        kernel = kernel / kernel.sum(1, keepdim=True)
        kernel = kernel.unsqueeze(1)
        self.filter(kernel, channels)  
    
    def vert_edge(self, channels, size):
        print("Detecting Vertical Edges...")
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
        self.filter(kernel, channels)
    
    def horiz_edge(self, channels, size):
        print("Detecting horizontal edges...")
        kernel = torch.zeros((size, size)).tolist()
        for i in range(size):
            for j in range(size):
                if i == (size // 2):
                    kernel[i][j] = 0
                elif i % 2 == 0:
                    if i < (size // 2):
                        if j <= (size // 2):
                            kernel[i][j] = -(2 * (j + 1))
                        else:
                            kernel[i][j] = -(abs(kernel[i][j-1]) - 2)
                    else:
                        if j <= (size // 2):
                            kernel[i][j] = (2 * (j + 1))
                        else:
                            kernel[i][j] = (kernel[i][j-1] - 2)
                else:
                    if i < (size // 2):
                        if j <= (size // 2):
                            kernel[i][j] = -(j + 1)
                        else:
                            kernel[i][j] = -(abs(kernel[i][j-1]) - 1)
                    else:
                        if j <= (size // 2):
                            kernel[i][j] = (j + 1)
                        else:
                            kernel[i][j] = (kernel[i][j-1] - 1)

        kernel = torch.tensor(kernel).unsqueeze(0).float()      
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.filter(kernel, channels)

    def sharpen(self, channels):
        print("Sharpening image...")
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]).unsqueeze(0).float()
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.filter(kernel, channels)
    
    def filter(self, kernel, channels):
        img = F.conv2d(self.img, kernel, groups=channels)
        img = img.permute(0, 2, 3, 1).squeeze(0).squeeze(0)
        img = img.detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        cv2.imshow('main', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = 'images/snowsper.png'
    my_blur = Filter(img)
    my_blur.sharpen(4)

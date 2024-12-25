from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class Filter:
    def __init__(self, img):
        # self.img = Image.open(img)
        self.img = np.array(img)
        self.img = torch.tensor(self.img).float()
        self.img = self.img.unsqueeze(0)
        # print(self.img.shape)
        self.img = self.img.permute(0, 3, 1, 2)
    
    def guassian_blur(self, channels:int, size:int, sigma:float) -> None:
        print("Applying Guassian Blur...")
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        x, y = np.meshgrid(x, y)

        kernel = (1 / (2*np.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2*sigma**2)))
        kernel = torch.tensor(kernel).float()
        kernel = kernel / kernel.sum(1, keepdim=True)
        kernel = kernel.unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return self.filter(kernel, channels, 'guassian blur')
    
    def default_blur(self, channels:int, size:int) -> None:
        print("Applying Default Blur...")
        kernel = torch.ones((channels, size, size)).float()
        kernel = kernel / kernel.sum(1, keepdim=True)
        kernel = kernel.unsqueeze(1)
        return self.filter(kernel, channels, 'default blur')  
    
    def vert_edge(self, channels:int, size:int) -> None:
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
        return self.filter(kernel, channels, 'verticle edge')
    
    def horiz_edge(self, channels:int, size:int) -> None:
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
        return self.filter(kernel, channels, 'horiz edge')

    def sharpen(self, channels:int) -> None:
        print("Sharpening image...")
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]).unsqueeze(0).float()
        kernel = kernel.repeat(channels, 1, 1, 1)
        return self.filter(kernel, channels, 'sharpen')
    
    def filter(self, kernel:torch.tensor, channels:int, type:str) -> None:
        img = F.conv2d(self.img, kernel, groups=channels, padding=1)

        if type == 'sharpen':
            img = torch.clamp(img, 0, 255)
            img = img.permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            img = img.detach().cpu().numpy()
        else:
            img = img.permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            img = img.detach().cpu().numpy()
            for i in range(img.shape[2]):
                channel = img[..., i]
                channel = (channel - channel.min()) / (channel.max() - channel.min()) * 255
                img[..., i] = channel

        img = img.astype(np.uint8)

        if img.shape[2] == 4: 
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        
        # cv2.imshow('main', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img



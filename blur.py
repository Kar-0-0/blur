from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class Blur:
    def __init__(self, img):
        self.img = Image.open('snowsper.png')
        self.img = np.array(self.img)
        self.img = torch.tensor(self.img).float()
        self.img = self.img.unsqueeze(0)
        self.img = self.img.permute(0, 3, 1, 2)
        
    def kernel_init(self, type, channels, size, sigma):

        if type == 'guassian':
            pass
        
        elif type == 'default':
            kernel = torch.ones((channels, size, size)).float()
            kernel = kernel / kernel.sum(1, keepdim=True)
            kernel = kernel.unsqueeze(0)
            return kernel

    def blur(self, in_channels, kernel_size):
        kernel = self.kernel_init('default', in_channels, kernel_size, 1)

        img = F.conv2d(self.img, kernel)
        img = img.permute(0, 1, 2, 3).squeeze(0).squeeze(0)
        img = img.detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        cv2.imshow('main', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = 'snowsper.png'
    my_blur = Blur(img)
    my_blur.blur(4, 15)






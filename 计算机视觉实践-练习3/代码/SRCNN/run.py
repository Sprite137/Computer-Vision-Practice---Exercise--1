import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os


def sr(img_file,model_file,zoom_factor,i):
    img = Image.open(img_file).convert('YCbCr')
    img = img.resize((int(img.size[0]*zoom_factor), int(img.size[1]*zoom_factor)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
    y, cb, cr = img.split()

    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load(model_file).to(device)
    input = input.to(device)

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
                                                                    # before converting the result in RGB
    out_img.save('SRCNN-master/sr{}/{}.png'.format(zoom_factor,i))
    
    
if __name__ == '__main__':
    for i,file in enumerate(os.listdir('Set5\original')):
        file_path = os.path.join('Set5\original',file)
        sr(file_path,'SRCNN-master\model\model_4x.pth',4,i)

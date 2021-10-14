# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:21:32 2021

@author: Master-Yi
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import dataset



'''空间注意力可视化'''
def visulize_spatial_attention(img_path, attention_mask, ratio=1, cmap="jet"):
    """
    img_path: image file path to load
    save_path: image file path to save
    attention_mask: 2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio: scaling factor to scale the output h and w
    cmap: attention style, default: "jet"
    """
    # print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    # scale表示放大或者缩小图片的比率
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.3, interpolation='nearest', cmap=cmap)
    plt.show()



if __name__ == "__main__":
    '''
    random_attention_mask = np.random.randn(16, 16) # h, w
    # print(random_attention_mask.shape) # (16, 16)
    img_path = 'slr-image/1.png'  # 图像路径
    visulize_spatial_attention(img_path=img_path, attention_mask=random_attention_mask)
    '''

    data_path='D:/Download/CSL_Continuous/color'
    dict_path='D:/Download/CSL_Continuous/dictionary.txt'
    corpus_path='D:/Download/CSL_Continuous/corpus.txt'

    transform = transforms.Compose([transforms.CenterCrop([672, 672]),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor()])

    trainset = dataset.CSL_Continuous(data_path=data_path,
                              dict_path=dict_path,
                              corpus_path=corpus_path,
                              train=True, transform=transform)


    for i in range(12):
        img = trainset[5][0][:,i,:,:].permute(1,2,0)
        # 更改颜色通道顺序：BGR -> RGB
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(img[:,:,(2,1,0)])
        plt.axis('off')
        plt.savefig('male_{}.png'.format(i+1), bbox_inches='tight', pad_inches=0)
    # plt.savefig('句子')

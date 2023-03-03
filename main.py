# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:06:36 2021

@author: Master-Yi
"""


'''这是程序入口'''

import time
import argparse
# 超算不能画图 import matplotlib.pyplot as plt

import solver


def parse_args():
    parser = argparse.ArgumentParser()
    # epoch和batch
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    # 数据集参数
    parser.add_argument("--crop_size", type=int, default=672)
    parser.add_argument("--img_size", type=int, default=224)

    # Adma参数
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)

    # 输入输出路径
    parser.add_argument("--data_path", type=str,
                        default='/project/gaoyi/CSL_Continuous/color')
    parser.add_argument("--dict_path", type=str,
                        default='/project/gaoyi/CSL_Continuous/dictionary.txt')
    parser.add_argument("--corpus_path", type=str,
                        default='/project/gaoyi/CSL_Continuous/corpus.txt')
    parser.add_argument("--out_path", type=str,
                        default='/project/gaoyi/mywork/out')

    config = parser.parse_args()
    return config


def epoch_out(opt):
    # import os
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []
    train_wers = []
    test_wers = []
    # counter = [i+1 for i in range(opt.n_epochs)]

    for e in range(opt.n_epochs):
        train_loss, train_acc, train_wer = solver.train(opt, e)
        train_losses.append(round(train_loss, 4))
        train_acces.append(round(train_acc, 4))
        train_wers.append(round(train_wer, 4))

        test_loss, test_acc, test_wer = solver.test(opt, e)
        test_losses.append(round(test_loss, 4))
        test_acces.append(round(test_acc, 4))
        test_wers.append(round(test_wer, 4))

    '''超算'''
    print("train_losses =", train_losses, flush = True)
    print("test_losses =", test_losses, flush = True)
    print("train_acces =", train_acces, flush = True)
    print("test_acces =", test_acces, flush = True)
    print("train_wers =", train_wers, flush = True)
    print("test_wers =", test_wers, flush = True)

    '''
    plt.plot(counter, train_losses, color='blue')
    plt.plot(counter, test_losses, color='green')
    plt.legend(['Train Loss', 'Test Loss'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(os.path.join(opt.out_path, 'loss.png'))

    plt.plot(counter, train_acces, color='blue')
    plt.plot(counter, test_acces, color='green')
    plt.legend(['Train Acc', 'Test Acc'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    # plt.show()
    plt.savefig(os.path.join(opt.out_path, 'acc.png'))

    plt.plot(counter, train_weres, color='blue')
    plt.plot(counter, test_weres, color='green')
    plt.legend(['Train Wer', 'Test Wer'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Wer')
    # plt.show()
    plt.savefig(os.path.join(opt.out_path, 'wer.png'))
    '''




if __name__ == '__main__':
    opt = parse_args()
    import torch

    start = time.perf_counter() # 开始时间

    print("Using CPU", flush = True)
    if torch.cuda.is_available():
        print("Update: Using GPU", flush = True)
        if torch.cuda.device_count() > 1:
            print("Update: Using {} GPUs".format(torch.cuda.device_count()), flush = True)
            print("\n")
    epoch_out(opt)

    end = time.perf_counter() # 结束时间
    total = end - start
    hours = total//3600
    minutes = total//60 - hours*60
    seconds = (total - hours*3600 - minutes*60) // 1

    print("总用时：{:n}小时 {:n}分钟 {:n}秒".format(hours, minutes, seconds), flush = True)

    # 直接测试
    # solver.test(opt, 1)
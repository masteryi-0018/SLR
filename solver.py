# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:24:18 2021

@author: Master-Yi
"""


import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset
from model import Seq2Seq


'''训练'''
def train(opt, epoch):
    # 参数
    data_path = opt.data_path
    dict_path = opt.dict_path
    corpus_path = opt.corpus_path

    batch_size = opt.batch_size
    crop_size = opt.crop_size
    img_size = opt.img_size
    lr = opt.lr
    b1 = opt.b1
    b2 = opt.b2

    # 数据集和变换
    transform = transforms.Compose([transforms.CenterCrop([crop_size, crop_size]),
                                    transforms.Resize([img_size, img_size]),
                                    transforms.ToTensor()])

    trainset = dataset.CSL_Continuous(data_path=data_path,
                                       dict_path=dict_path,
                                       corpus_path=corpus_path,
                                       train=True, transform=transform)

    '''超算'''
    # num_workers=16, pin_memory=True, drop_last=True
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, drop_last=True)


    # 模型，如果epoch不为0，就加载已经训练过的ckpt
    len_dict = trainset[0][3]
    model = Seq2Seq(len_dict=len_dict)

    model.train()

    # 损失函数，优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr, (b1, b2))

    # 设定gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if epoch != 0:
        ckpt = os.path.join(opt.out_path, 'model.pth')

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)
            criterion.to(device)

            # 多块gpu只转移模型
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1,2,3])
                model.to(device)

        model.load_state_dict(torch.load(ckpt))

    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)
            criterion.to(device)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1,2,3])
                model.to(device)



    # 开始训练
    n_batchs = len(trainloader)
    running_loss = 0.0
    losses = []
    all_trg = [] # 目标和预测的向量，用来计算
    all_pred = []
    all_wer = []

    print('--Started Training And Testing--', flush = True)
    for batch_idx, data in enumerate(trainloader):
        imgs, target = data[0], data[1]
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # forward
        outputs = model(imgs, target, 0.5)
        # 换回来
        outputs = outputs.permute(1,0,2)

        # target: (batch_size, len_label)
        # outputs: (len_label, batch_size, output_dim)
        # skip sos 然后计算损失
        output_dim = outputs.shape[-1]
        # outputs torch.Size([32, 9, 507]) 507为每个词在字典中的概率

        outputs = outputs[1:].reshape(-1, output_dim)
        # 112, 500, 7*16=112
        target = target.permute(1,0)[1:].reshape(-1)
        # target也去掉sos，交换维度就是为了方便去掉sos
        # 变成112大小，正好符合交叉熵，这里是看500个词语里面预测正确多少


        # compute the loss
        loss = criterion(outputs, target)
        losses.append(loss.item())
        running_loss += loss.item()
        # print(loss.item()) 是一个数值

        # backward & optimize
        loss.backward()
        # 防止过拟合
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()


        '''指标'''
        # 对于batch
        # compute the accuracy
        # 找出outputs的最大值对应的词汇，后面的1代表就是最大值对应的序号
        prediction = torch.max(outputs, 1)[1]
        # 为什么squeeze，没有效果 prediction torch.Size([112])
        # 每个词都要从字典里找出最大的那个，判断是否相同
        score = accuracy_score(target.cpu().data.squeeze().numpy(),
                               prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)
        # target and pred: batch*(len_label-1)

        # compute wer，变回list，长度为batch，每个项目中有len_label个
        # prediction: ((len_label-1)*batch_size)
        # target: ((len_label-1)*batch_size)
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []
        for i in range(batch_size):
            # add mask(remove padding, sos, eos)
            # 去掉每个batch的标志词，然后计算wer
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))
        # print(wers) batch个wer
        all_wer.extend(wers)


        # 打印
        # if batch_idx % sam == 0:
            # print("epoch:%d/%d, batch:%d/%d, loss:%.4f, acc:%.2f, wer:%.2f" %
                  # (epoch+1, opt.n_epochs, batch_idx+1, n_batchs,
                  #  running_loss/(batch_idx+1), score*100, sum(wers)/len(wers)), flush = True)


    # 对于epoch
    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses) # losses是列表，取所有batch的平均值作为epoch的loss
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    # dim=0 没有增加维度
    # print(all_trg.shape, all_pred.shape) torch.Size([320]) torch.Size([320])

    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(),
                                  all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)

    # 训练结束
    # print('--Finished Training--', flush = True)
    print("epoch:%d, loss:%.2f, acc:%.2f, wer:%.2f" %
          (epoch+1, training_loss, training_acc*100, training_wer), flush = True)

    os.makedirs(opt.out_path, exist_ok=True)
    pth_path = os.path.join(opt.out_path, 'model.pth')
    '''
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        # torch.save(model.module.state_dict(), pth_path)  # 修改成这一行
        torch.save(model.state_dict(), pth_path)
    else:
        torch.save(model.cpu().state_dict(), pth_path)
    '''
    torch.save(model.state_dict(), pth_path)
    # print('--Model Saved--\n', flush = True)

    return training_loss, training_acc, training_wer


'''验证'''
def test(opt, epoch):
    # 利用原始.pth模型进行前向推理之前，一定要先进行model.eval()操作
    # 不启用 BatchNormalization 和 Dropout

    # 参数
    data_path = opt.data_path
    dict_path = opt.dict_path
    corpus_path = opt.corpus_path

    batch_size = opt.batch_size
    crop_size = opt.crop_size
    img_size = opt.img_size

    # 数据
    transform = transforms.Compose([transforms.CenterCrop([crop_size, crop_size]),
                                    transforms.Resize([img_size, img_size]),
                                    transforms.ToTensor()])

    testset = dataset.CSL_Continuous(data_path=data_path,
                                       dict_path=dict_path,
                                       corpus_path=corpus_path,
                                       train=False, transform=transform)

    '''超算'''
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True,
                            num_workers=16, pin_memory=True, drop_last=True)


    criterion = nn.CrossEntropyLoss()

    # 加载训练好的模型
    model_path = os.path.join(opt.out_path, 'model.pth')
    len_dict = testset[0][3]
    model = Seq2Seq(len_dict = len_dict)

    # gpu设定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)
        criterion.to(device)

        # 多块gpu只转移模型
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    n_iters = len(testloader)
    running_loss = 0.0
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    # print('--Started Testing--', flush = True)

    # 验证阶段不更新模型参数，同样放在gpu
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            imgs, target = data[0], data[1]
            imgs = imgs.to(device)
            target = target.to(device)

            # forward 这里不加强化学习的0.5 即只能获取上一次预测的值
            outputs_o = model(imgs, target, 0)
            outputs_o = outputs_o.permute(1,0,2)

            # target: (batch_size, len_label)
            # outputs: (len_label, batch_size, output_dim)
            # skip sos 然后计算损失
            output_dim = outputs_o.shape[-1]

            outputs = outputs_o[1:].reshape(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)
            # target也去掉sos，交换维度就是为了方便去掉sos
            # 变成112大小，正好符合交叉熵，这里是看500个词语里面预测正确多少

            # compute the loss
            loss = criterion(outputs, target)
            losses.append(loss.item())
            running_loss += loss.item()

            '''指标'''
            # compute the accuracy
            # 找出outputs的最大值对应的词汇，后面的1代表就是最大值对应的序号
            prediction = torch.max(outputs, 1)[1]
            # 为什么squeeze，没有效果 prediction torch.Size([112])
            # 每个词都要从字典里找出最大的那个，判断是否相同
            score = accuracy_score(target.cpu().data.squeeze().numpy(),
                                   prediction.cpu().data.squeeze().numpy())
            all_trg.extend(target)
            all_pred.extend(prediction)

            # compute wer，变回list，长度为batch，每个项目中有len_label个
            # prediction: ((len_label-1)*batch_size)
            # target: ((len_label-1)*batch_size)
            prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
            target = target.view(-1, batch_size).permute(1,0).tolist()

            wers = []
            for i in range(batch_size):
                # add mask(remove padding, sos, eos)
                # 去掉每个batch的标志词，然后计算wer
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))

            all_wer.extend(wers)

            # 打印
            # if idx % (sam/4) == 0:
                # print("epoch:%d/%d, batch:%d/%d, loss:%.4f, acc:%.2f, wer:%.2f" %
                      # (epoch+1, opt.n_epochs, idx+1, n_iters,
                      #  running_loss/(i+1), score*100, sum(wers)/len(wers)), flush = True)

            '''超算'''











        # Compute the average loss & accuracy
        # 对于epoch
        testing_loss = sum(losses)/len(losses)
        all_trg = torch.stack(all_trg, dim=0)
        all_pred = torch.stack(all_pred, dim=0)

        testing_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(),
                                      all_pred.cpu().data.squeeze().numpy())
        testing_wer = sum(all_wer)/len(all_wer)


    # print('--Finished Testing--', flush = True)
    print("epoch:%d, loss:%.2f, acc:%.2f, wer:%.2f\n" %
          (epoch+1, testing_loss, testing_acc*100, testing_wer), flush = True)

    return testing_loss, testing_acc, testing_wer



def wer(r, h):
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)


    return float(d[len(r)][len(h)]) / len(r) * 100
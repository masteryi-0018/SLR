# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:00:17 2021

@author: Master-Yi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

# import math
from functools import partial


# encoder负责将输入序列压缩成指定长度的向量，这个向量就可以看成是这个序列的语义，这个过程称为编码
class Encoder(nn.Module):
    def __init__(self, lstm_hidden_size=512, arch="resnet"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # network architecture

        # 加载预训练模型，用预训练模型的参数来初始化
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=False)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=False)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=False)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=False)
        else:
            resnet = ResNet()

        '''使用3d resnet'''
        # self.res3d = generate_model(34)
        # delete the last fc layer
        # [:-1]表示选取1到-1，即倒数第一个（不含），fc本来是512-1000
        # modules = list(resnet.children())[:-1]
        # self.resnet = nn.Sequential(*modules)
        self.resnet = resnet

        # 最后一层换成lstm
        self.lstm = nn.LSTM(
            # input_size=resnet.fc.in_features, # 512
            input_size=512,
            hidden_size=self.lstm_hidden_size, # 512
            # 如果是True，则input为(batch, seq, input_size)，默认值为：False(seq_len, batch, input_size)
            # num_layers：堆叠LSTM的层数，默认值为1
            batch_first=True
            )

        # attn encoder
        # 时间注意力
        self.attn = LSTMAttentionBlock(hidden_size=512)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        # 对每一张图片进行
        for t in range(x.size(2)):
            # with torch.no_grad()
            out = self.resnet(x[:, :, t, :, :])
            # 4维
            # print(out.shape) torch.Size([16, 512, 1, 1])
            # 有avgpool - AdaptiveAvgPool2d(output_size=(1, 1)),
            out = out.view(out.size(0), -1)
            # print(out.shape) torch.Size([16, 512])
            cnn_embed_seq.append(out)
            # 一共增加了t个seq，也就是lstm中的一个标签对应的词向量长度

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print('resnet输出：', cnn_embed_seq.shape)
        # print(cnn_embed_seq.shape) torch.Size([12, 32, 512])
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)
        # print(cnn_embed_seq.shape) torch.Size([32, 12, 512])


        '''改用3d resnet'''
        # out = self.res3d(x)
        # cnn_embed_seq = out.permute(0,2,1)

        # 原本的shape: torch.Size([32, 12, 512])
        # 新的shape: torch.Size([2, 12, 512])


        # LSTM
        # use faster code paths

        # 如果不加这句，多卡跑模型会出错
        self.lstm.flatten_parameters()
        # 前向计算过程,这里不传入h_0和c_0则会默认初始化
        out, (h_n, c_n) = self.lstm(cnn_embed_seq)
        # h0~hn:上一刻时间产生的中间值与当前时刻的输入共同产生的状态
        # c0~cn:开关，决定每个神经元的隐藏状态值是否会影响下一时刻神经元的处理

        '''对lstm的out的说明'''
        # tensor containing the output features (h_t) from the last layer of the LSTM, for each t
        # 这里的h_t是网络的隐含状态，和上面的h_n不一样

        # encoder也可以加embedding

        # output：(seq_len, batch, num_directions * hidden_size)
        # h_n：(num_layers * num_directions, batch, hidden_size)
        # c_n：(num_layers * num_directions, batch, hidden_size)

        # num_layers * num_directions = 1
        # torch.Size([32, 12, 512]) torch.Size([32, 512]) torch.Size([32, 512])

        # 加入encoder的注意力
        out = self.attn(out)
        # 加入注意力 torch.Size([32, 512])
        # 未加入注意力 torch.Size([32, 12, 512])

        return out, (h_n.squeeze(0), c_n.squeeze(0))


# Decoder的每一时刻的输入为Eecoder输出的c 和Decoder前一时刻解码的输出s(i-1)，
# 还有前一时刻预测的词的向量e(i-1)
# 将encoder得到的语义变量作为初始状态输入到decoder的rnn中，得到输出序列
# 语义向量C只作为初始状态参与运算，后面的运算都与语义向量C无关
# decoder处理方式还有另外一种，就是语义向量C参与了序列所有时刻的运算
class Decoder(nn.Module):
    def __init__(self, len_dict, emb_dim=256, dropout=0.5):
        super(Decoder, self).__init__()
        enc_hid_dim=512 # 和encoder最终输出的dim，即lstm_hidden_size
        dec_hid_dim=512

        self.len_dict = len_dict
        # 拿到字典的长度，对其进行相应的编码
        self.embedding = nn.Embedding(len_dict, emb_dim)
        # 自动学习每个词向量对应的w权重
        # num_embeddings - 词嵌入字典大小，即一个字典里要有多少个词
        # embedding_dim - 嵌入向量的维度，即用多少维来表示一个符号

        self.lstm = nn.LSTM(emb_dim+enc_hid_dim, dec_hid_dim)
        # 这里就是256 + 512 表示每个词是多少维的向量，这里的词是emb+con的结果

        self.fc = nn.Linear(emb_dim+enc_hid_dim+dec_hid_dim, len_dict)
        # 因为输入为emb lstm_out 还加了一个隐藏层h_n，最终到字典长度

        self.dropout = nn.Dropout(dropout)

        self.max = nn.Softmax(dim=1) # 每一行之和为0

    def forward(self, in_put, h_n, cell, context):
        # in_put(batch_size): last prediction
        # hidden(batch_size, dec_hid_dim): decoder last hidden state
        # cell(batch_size, dec_hid_dim): decoder last cell state
        # context(batch_size, enc_hid_dim): context vector
        # print(input.shape, hidden.shape, cell.shape, context.shape)
        # expand dim to (1, batch_size)

        # 输入为一组batch的列的值 torch.Size([16])，这里加一维 torch.Size([1, 16])
        in_put = in_put.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        # RNN的每一步要输入每个样例的一个单词
        embedded = self.embedding(in_put)
        '''embeded输出的维度是[1,16,256],这就代表对于输入维度为1x16的词，
        每个词都被映射成了一个256维的向量，这里多了最后的维度emb_dim=256'''

        embedded = self.dropout(embedded)

        # rnn_input(1, batch_size, emb_dim+enc_hide_dim): concat embedded and context

        # 对句子级的标签也cat到一起 emb.shape=[1 16 256] con.un.shape=[1 16 512]
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        # 输入不仅有context 还有emb之后的结果


        # 如果不加这句，多卡跑模型会出错
        self.lstm.flatten_parameters()
        # output(seq_len, batch, num_directions * hidden_size)
        # hidden(num_layers * num_directions, batch, hidden_size)
        output, (hidden, cell) = self.lstm(rnn_input, (h_n.unsqueeze(0), cell.unsqueeze(0)))

        '''decoder的lstm出来，一般会加softmax'''
        # hidden(batch_size, dec_hid_dim)
        # cell(batch_size, dec_hid_dim)
        # embedded(1, batch_size, emb_dim)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)

        # prediction - 16*256, 16*512, hidden是重要的信息
        # emb context hidden(lstm的)，就直接预测了？
        # 确实，将每次的hidden与其他2个通过线性层，就得到了预测向量
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))
        # 从batch*加和 变成 batch*len_dict

        # 为什么不加softmax
        # 加
        # prediction = self.max(prediction)
        # 预测的值是什么 torch.Size([16, 500])
        # 每个样本都变成了500(len_dict)维的向量
        return prediction, (hidden, cell)

# many to many 的 rnn 输入和输出序列不等长，这种模型便是seq2seq模型
# 在训练阶段可能对rnn的输出不处理，直接用target的序列作为下时刻的输入。
# 而预测阶段会将rnn的输出当成是下一时刻的输入
class Seq2Seq(nn.Module):
    def __init__(self, len_dict):
        super(Seq2Seq, self).__init__()

        self.len_dict = len_dict
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder()
        self.decoder = Decoder(len_dict=len_dict)


    def forward(self, imgs, target, teacher_forcing_ratio=0.5):

        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, len_label)
        batch_size = imgs.shape[0]

        # 标签长度(有几个单词)
        len_label = target.shape[1]
        # 字典长度
        len_dict = self.len_dict

        # 超算 torch.Size([36, 8, 507])
        # print(len_label, batch_size) # 9 8 batch分配到每个gpu上就是8
        # tensor to store decoder outputs (9, 32, 507)
        outputs = torch.zeros([len_label, batch_size, len_dict], device=self.device).permute(1,0,2)
        # print(torch.cuda.current_device(), outputs.shape) # 这里每个outputs都是torch.Size([9, 8, 507]) permute之前
        # 先交换再拼接，torch.Size([32, 9, 507]) 每个都是8 9 507

        # seq_len就是帧数=12  hidden_size=512
        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, (hidden, cell) = self.encoder(imgs)
        ''' encoder_outputs context 参与了所有的decoder'''

        # compute context vector
        # 对帧数求平均，每个样本都是看作一个完整句子 16, 12, 512 -> 16, 512
        # (上下文向量)context vector 来源是隐含状态h
        # 因为它编码了整个文本序列。这个上下文向量被用作解码器的初始隐藏状态。
        # context = encoder_outputs.mean(dim=1)
        # 添加注意力后需要修改为以下表达
        context = encoder_outputs

        # first input to the decoder is the <sos> tokens
        in_put = target[:,0]
        # 每次取出一个batch的标签最前面的，先取出sos，后面循环
        # 对每个单词操作
        for t in range(1, len_label):
            # decode
            # in_put.shape torch.Size([batch_size])
            # in_put是第一个词，hidden cell context是encoder的输出，这里给了每个decoder
            output, (hidden, cell) = self.decoder(in_put, hidden, cell, context)

            # store prediction
            # 对每个单词解码，预测值放入outputs
            # output.shape torch.Size([32, 507]) 这是一个batch的数据
            outputs[:,t,:] = output

            # decide whether to do teacher foring
            # teacher foring 值越低监督越轻
            # 设为0 一直是False，即decoder只用自己预测的最大值进行预测
            teacher_force = random.random() < teacher_forcing_ratio
            '''训练过程中，使用要解码的序列作为输入进行训练，它帮助模型加速收敛
            但是在inference阶段是不能使用的，因为你不知道要预测的序列是个啥'''
            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            # 如果满足条件，就是下一个 target[:,t] 否则是 top1
            # 测试时一直是top1
            in_put = target[:,t] if teacher_force else top1
            # 这里转变为下一组

        # 输入图像和句子，输出标签长度的decoder outputs [9, 32, 507]
        # lstm 不是batch first

        # 每个运行完还是torch.Size([8, 9, 507]) 之后再放进同一个outputs，再换回来
        # outputs = outputs.permute(1,0,2) # [9, 32, 507]
        # print(torch.cuda.current_device(), outputs.shape)
        # 这里不将其换回来，因为默认还是对dim=0进行batch拼接，所以等到模型输出后再操作
        return outputs



'''原始resnet2d'''
class ResidualBlock(nn.Module):
    """实现一个残差块"""
    def __init__(self,inchannel,outchannel,stride = 1,shortcut = None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False), # 这个卷积操作是不会改变w h的
            nn.BatchNorm2d(outchannel)
            )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out+=residual
        return F.relu(out)


class ResNet(nn.Module):
    """实现主reset"""
    def __init__(self,num_class=1000):
        super().__init__()
        '''这里的层的定义应当和forward中的一致！'''
        # 前面几层普通卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
            )

        # 这是通道注意力的进入通道数
        # self.inplanes = 64
        # 空间注意力
        self.sa = SpatialAttention()
        # 通道注意力
        self.ca = ChannelAttention(64)

        # 重复layer，每个layer都包含多个残差块 其中第一个残差会修改w和c，其他的残差块等量变换
        # 经过第一个残差块后大小为 w-1/s +1 （每个残差块包括left和right，而left的k = 3 p = 1，right的shortcut k=1，p=0）
        self.layer1 = self._make_layer(64,128,3) # s默认是1 ,所以经过layer1后只有channle变了
        self.layer2 = self._make_layer(128,256,4,stride=2) # w-1/s +1
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512,num_class)


    def _make_layer(self,inchannel,outchannel,block_num,stride = 1):

        # 刚开始两个cahnnle可能不同，所以right通过shortcut把通道也变为outchannel
        shortcut = nn.Sequential(
            # 之所以这里的k = 1是因为，我们在ResidualBlock中的k =3,p=1所以最后得到的大小为(w+2-3/s +1)
            # 即(w-1 /s +1)，而这里的w = (w +2p-f)/s +1 所以2p -f = -1 如果p = 0则f = 1
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))

        # 之后的cahnnle同并且 w h也同，而经过ResidualBloc其w h不变，
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.pre(x)

        x = self.sa(x) * x
        x = self.ca(x) * x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = F.avg_pool2d(x, 7)
        # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        # x = x.view(x.size(0),-1)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return x



'''通道注意力'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #print(avg_out.shape, max_out.shape)
        out = avg_out + max_out

        return self.sigmoid(out)



'''空间注意力'''
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)



'''时间注意力'''
class LSTMAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)

        '''注意力的两个矩阵相乘'''
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector




if __name__ == '__main__':

    '''假设 t=12 len_dict=507 img_size=64'''
    imgs = torch.randn(2, 3, 12, 224, 224)
    target = torch.LongTensor(2, 9).random_(0, 507)

    '''
    # test encoder
    encoder = Encoder()
    out, hc = encoder(imgs)
    print(out.shape, hc[0].shape, hc[1].shape)

    # test decoder
    decoder = Decoder(len_dict=500)
    in_put = torch.LongTensor(16).random_(0, 500)
    hidden = torch.randn(16, 512)
    cell = torch.randn(16, 512)
    context = torch.randn(16, 512)
    out, hc = decoder(in_put, hidden, cell, context)
    print(out.shape, hc[0].shape, hc[1].shape)

    # test ca
    ca = ResNet()
    img = torch.randn(2,3,224,224)
    out = ca(img)
    print("原始resnet：", out.shape)
    '''

    # test seq2seq
    seq2seq = Seq2Seq(len_dict=507)
    out = seq2seq(imgs, target)
    print("最终输出形状：", out.shape)
# SLR

## 数据集

- 数据集来自：
  http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html
  
- 数据处理：
  crop至`672*672`，再resize成`224*224`
  
- 抽帧

## 超参数

- 模型：resnet + lstm
- 损失：cross entropy
- 优化器：adam
- 学习率：None

## 实验过程

- 超参数实验
  1. 不同网络深度的影响
    - resnet18
    - resnet50
    - resnet152

  2. 不同关键帧的影响
    - key frame = 8
    - key frame = 12
    - key frame = 16

  3. 不同teacher forcing的影响
    - teacher forcing = 0.2
    - teacher forcing = 0.5
    - teacher forcing = 0.8

- 消融实验
  1. spatial attention
  2. spatial attention + channel attention
  3. spatial attention + channel attention + time attention

## 实验结果

待补充

## 文件结构

```
README.md     -readme
dataset.py    -读取数据集
main.py       -程序入口
model.py      -模型
plot.py       -画图
resnet3d.py   -resnet3d
solver.py     -训练与验证
visulize.py   -可视化
```


# coding: utf-8
from models.resnet_arch import ResNet, BasicBlock, Bottleneck
from models.faresnet_model import FaResNet
from dataloader import data_load

import numpy as np
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os

# 가중치 초기화
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

config = {
    'epochs': 10,    # => test roc auc : 0.7479618212865566
    'batch_size': 32,
    'root': 'D:/Mtg-jamendo-dataset/melspecs',
    'tag_path': './tags',
    'model_save_path': './trained/faresnet/'
}

layer_index_total = 0
first_RUN = True


def run():
    global config
    # rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like nets.
    # rho = 5
    # mixup = 1

    # configs/cp_resnet.json 참고
    batch_size = config['batch_size']     # 128
    epochs = config['epochs']
    model_save_path = config['model_save_path']

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # 데이터 로더 만들기
    train_dataset, val_dataset, test_dataset = data_load(root=config['root'], tag=config['tag_path'], annotation=False)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = FaResNet(BasicBlock)
    print(model)

    checkpoint = ModelCheckpoint(
        dirpath=model_save_path
    )
    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)



if __name__ == '__main__':
    run()
















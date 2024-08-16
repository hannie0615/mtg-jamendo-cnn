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
    'epochs': 200,    # => test roc auc : 0.7479618212865566
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/faresnet/',
    'mode': 'TEST-', # TRAIN or TEST
}

layer_index_total = 0
first_RUN = True


def run():
    global config
    # rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like nets.
    # rho = 5
    # mixup = 1

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

    model = FaResNet(block=BasicBlock)
    # print(model)
    # model = FaResNet.load_from_checkpoint('trained/faresnet/faresnet-epoch=10-val_loss=6.12.ckpt')

    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="faresnet-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])
    if config['mode'] == 'TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode'] == 'TEST':
        trained_model_path = os.path.join(model_save_path, 'crnn-epoch=136-val_loss=6.45.ckpt')
        model = FaResNet.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")

    ## model save path 안에 있는 모든 ckpt 파일을 한번에 test 하기(필요한 경우만 사용)
    # file_list = os.listdir(model_save_path)
    #
    # for tf in file_list:
    #     path = os.path.join(model_save_path, tf)
    #     etf = FaResNet.load_from_checkpoint(path)
    #     print(path)
    #     trainer.test(etf, test_dataloader)


if __name__ == '__main__':
    run()
















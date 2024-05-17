from models.ensemble1_model import MyEnsemble1
from models.faresnet_model import FaResNet
from models.resnet_arch import BasicBlock
from models.crnn_model import CRNN_esb
from dataloader import data_load

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

config = {
    'epochs': 10,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/ensemble3/'
}

def run():
    print('start')
    global config

    batch_size = config['batch_size']  # 32 그대로
    epochs = config['epochs']
    model_save_path = config['model_save_path']

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 만들기
    train_data, val_data, test_data = data_load(root=config['root'], tag=config['tag_path'])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # 모델 초기화
    model1 = FaResNet(BasicBlock)
    model2 = CRNN_esb()

    ensemble_model = MyEnsemble1(model1, model2)

    # 모델 앙상블
    print(ensemble_model)

    checkpoint = ModelCheckpoint(
        dirpath=model_save_path,
    )
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint])

    trainer.fit(ensemble_model, train_dataloader, val_dataloader)
    trainer.test(ensemble_model, test_dataloader)


if __name__ == '__main__':
    run()

from models.ensemble3_model import MyEnsemble3
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
    'epochs': 100,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/ensemble3/',
    'mode':'TEST'
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
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # 모델 초기화
    model1 = FaResNet(block=BasicBlock, num_classes=56)
    model2 = CRNN_esb()

    model = MyEnsemble3(modelA=model1, modelB=model2)

    # 모델 앙상블
    # print(ensemble_model)

    # Save model
    checkpoint = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="faresnet+crnn-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint])

    if config['mode'] == 'TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode'] == 'TEST':
        trained_model_path = os.path.join(model_save_path, 'faresnet+crnn-epoch=149-val_loss=0.13.ckpt')
        model = MyEnsemble3.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")


if __name__ == '__main__':
    run()

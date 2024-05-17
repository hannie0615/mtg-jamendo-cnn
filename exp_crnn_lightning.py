# 라이브러리 임포트
from models.crnn_model import CRNN
from dataloader import data_load
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


config = {
    'epochs': 10,    # 200
    'batch_size': 8,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './models/crnn/'
}


initialized = False  # TODO: Find a better way to do this
trial_counter = 0

def run():
    print('start')
    global config

    batch_size = config['batch_size']
    epochs = config['epochs']
    model_save_path = config['model_save_path']

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # 데이터 로더 만들기
    train_data, val_data, test_data = data_load(root=config['root'], tag=config['tag_path'])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # 모델 선언하기
    model = CRNN(num_class=56)
    checkpoint = ModelCheckpoint(dirpath=model_save_path)

    # trainer 선언하기
    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])

    # print(model)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    # plt.figure(figsize=(8, 4))
    # plt.plot(train_loss)
    # plt.plot(val_loss)
    # plt.show()
    # plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()







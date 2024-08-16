from models.ensemble2_model import MyEnsemble2
from models.vggnet_model import VggNet
from models.crnn_model import CRNN_esb

import os
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from dataloader import data_load

config = {
    'epochs': 40,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/ensemble2/',
    'mode':'TEST'
}

def run():
    print('start')
    global config

    batch_size = config['batch_size']
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
    model1 = VggNet()
    model2 = CRNN_esb()

    # 모델 앙상블
    model = MyEnsemble2(model1, model2)
    # ensemble_model = MyEnsemble2.load_from_checkpoint('trained/ensemble2/vggnet+crnn-epoch=102-val_loss=6.52.ckpt')


    # Save model
    checkpoint = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="vggnet+crnn-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint])

    if config['mode'] == 'TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode'] == 'TEST':
        trained_model_path = os.path.join(model_save_path, 'vggnet+crnn-epoch=164-val_loss=6.63.ckpt')
        model = MyEnsemble2.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")

if __name__ == '__main__':
    run()

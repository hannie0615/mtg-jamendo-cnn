from models.ensemble1_model import MyEnsemble1
from models.resnet_model import ResNet34
from models.crnn_model import CRNN_esb

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from dataloader import data_load

config = {
    'epochs': 10,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/ensemble1/'
}


def run():
    global config
    print('start')

    batch_size = config['batch_size']     # 32 그대로
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
    model1 = ResNet34()
    model2 = CRNN_esb()

    # 모델 앙상블
    eensemble_model = MyEnsemble1()
    # ensemble_model = MyEnsemble1.load_from_checkpoint('trained/ensemble1/resnet+crnn-epoch=141-val_loss=0.13.ckpt')

    # Save model
    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="resnet+crnn-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint])

    trainer.fit(ensemble_model, train_dataloader, val_dataloader)
    trainer.test(ensemble_model, test_dataloader)



if __name__ == '__main__':
    run()

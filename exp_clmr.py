from models.baseline_model import BaseModel
from models.CLMR_model import CLMRBaseModel, CLMR_CNN, CLMR_classifier

import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from dataloader import data_load, data_clmr_load

config = {
    'epochs': 10,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'root2' : 'D:/Dcase-task1/mel',
    'tag_path': './tags',
    'model_save_path': './trained/clmr/mlp'
}   # tag path 는 그냥 tag path.

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
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CLMRBaseModel(batch_size=batch_size)
    print(model)

    # 체크 포인트 선언 ModelCheckpoint -> 왜 _init_ 오류가..?
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path
    )

    trainer = Trainer(max_epochs=epochs, default_root_dir=model_save_path)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

def cnn_run():
    global config
    batch_size = config['batch_size']
    epochs = config['epochs']
    model_save_path = config['model_save_path']

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # 데이터 로더 만들기
    train_data = data_clmr_load(root=config['root2'])

    rest = len(train_data) % 32
    train_data = train_data[:-rest]
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    model = CLMR_CNN(batch_size=batch_size)
    print(model)

    # 체크 포인트 선언 ModelCheckpoint -> 왜 _init_ 오류가..?
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path
    )
    trainer = Trainer(max_epochs=epochs, default_root_dir=model_save_path)
    trainer.fit(model, train_dataloader)

def mlp_run():
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
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CLMR_classifier(batch_size=batch_size)
    print(model)

    # 체크 포인트 선언 ModelCheckpoint -> 왜 _init_ 오류가..?
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path
    )

    trainer = Trainer(max_epochs=epochs, default_root_dir=model_save_path)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # cnn_run() # step 1 : cnn (epoch 200)
    mlp_run() # step 2 : classifier




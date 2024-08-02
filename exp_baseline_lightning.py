from models.baseline_model import BaseModel
from dataloader import data_load

import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


config = {
    'epochs': 200,    # 200
    'batch_size': 32,
    'root': './data',   # mel data path
    'tag_path': './tags',
    'model_save_path': './trained/baseline/'
}

def run():
    print('start')
    global config

    batch_size = config['batch_size']
    epochs = config['epochs']
    model_save_path = config['model_save_path']

    # 모델 저장 경로 만들기
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # 데이터 로더 만들기
    train_data, val_data, test_data = data_load(root=config['root'], tag=config['tag_path'])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4)

    model = BaseModel()
    print(model)

    # 체크 포인트 선언 ModelCheckpoint -> 왜 _init_ 오류?
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path
    )

    trainer = Trainer(max_epochs=epochs, default_root_dir=model_save_path)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()




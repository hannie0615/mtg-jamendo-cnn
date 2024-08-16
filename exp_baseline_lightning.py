from models.baseline_model import BaseModel
from dataloader import data_load

import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


config = {
    'epochs': 200,    # epoch(default:200))
    'batch_size': 32,   # batch size
    'root': './data/melspecs_5',   # mel data path
    'tag_path': './tags',           # tag data path
    'model_save_path': './trained/baseline/',   # ckpt model save path
    'mode': 'TEST',                 # mode : 'TRAIN' or 'TEST'
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

    # 데이터셋 만들기
    train_data, val_data, test_data = data_load(root=config['root'], tag=config['tag_path'])

    # 데이터로더 만들기
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4)

    # 모델 선언
    model = BaseModel()
    print(model)

    # checkpoint
    checkpoint = ModelCheckpoint(
        save_top_k=1,   # 상위 1개 모델을 저장(val loss 기준으로 가장 minimum한 값)
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="baseline-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])

    if config['mode']=='TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode']=='TEST':
        trained_model_path = os.path.join(model_save_path, 'baseline-epoch=179-val_loss=6.28.ckpt')
        model = BaseModel.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")

    ## model save path 안에 있는 모든 ckpt 파일을 한번에 test 하기(필요한 경우만 사용)
    # file_list = os.listdir(model_save_path)
    #
    # for tf in file_list:
    #     path = os.path.join(model_save_path, tf)
    #     etf = BaseModel.load_from_checkpoint(path)
    #     print(path)
    #     trainer.test(etf, test_dataloader)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()




# 라이브러리 임포트
from models.vggnet_model import VggNet
from dataloader import data_load
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

config = {
    'epochs': 200,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/vggnet/',
    'mode': 'TEST', # TRAIN or TEST
}

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
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint = ModelCheckpoint(
        save_top_k=1, # 상위 1개 저장
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="vggnet-{epoch:02d}-{val_loss:.2f}",
    )
    model = VggNet()
    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])


    if config['mode'] == 'TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode'] == 'TEST':
        trained_model_path = os.path.join(model_save_path, 'vggnet-epoch=162-val_loss=0.12.ckpt')
        model = VggNet.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()







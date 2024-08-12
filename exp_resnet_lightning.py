# 라이브러리 임포트
from models.resnet_model import ResNet34

import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from dataloader import data_load


config = {
    'epochs': 10,    # 200
    'batch_size': 32,
    'lr': 1e-4,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/resnet/'
}

def run():
    global config
    print('start')

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

    model = ResNet34()
    print(model)
    ## checkpoint 있을 때
    # model = Network.load_from_checkpoint('./models/resnet/epoch=199-step=62200.ckpt')
    # print(model)

    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="resnet-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint])  # default_root_dir='./models/',

    model = ResNet34()
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()









from models.ensemble4_model import MyEnsemble4
from models.faresnet_model import FaResNet
from models.resnet_arch import BasicBlock
from models.resnet_model import ResNet34
from models.vggnet_model import VggNet
from models.crnn_model import CRNN_esb
from dataloader import data_load
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


config = {
    'epochs': 200,    # 200
    'batch_size': 32,
    'root': './data/melspecs_5',
    'tag_path': './tags',
    'model_save_path': './trained/ensemble4/',
    'mode':'TEST'
}


def run():
    print('start')
    global config

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
    model1 = FaResNet(block=BasicBlock, num_classes=56)
    model2 = ResNet34()
    model3 = VggNet()
    model4 = CRNN_esb()

    ensemble_model = MyEnsemble4(model1, model2, model3, model4)

    # 모델 앙상블
    print(ensemble_model)

    # Save model
    checkpoint = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="everything-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint])

    if config['mode'] == 'TRAIN':
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

    elif config['mode'] == 'TEST':
        trained_model_path = os.path.join(model_save_path, 'everything-epoch=130-val_loss=0.13.ckpt')
        model = MyEnsemble4.load_from_checkpoint(trained_model_path)
        trainer.test(model, test_dataloader)

    else:
        print("check your MODE")

    ## model save path 안에 있는 모든 ckpt 파일을 한번에 test 하기(필요한 경우만 사용)
    # file_list = os.listdir(model_save_path)
    #
    # for tf in file_list:
    #     path = os.path.join(model_save_path, tf)
    #     etf = MyEnsemble4.load_from_checkpoint(path)
    #     print(path)
    #     trainer.test(etf, test_dataloader)


if __name__ == '__main__':
    run()

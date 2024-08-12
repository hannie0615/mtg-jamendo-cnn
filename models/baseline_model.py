import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

roc_aucs=[]
class BaseModel(pl.LightningModule):
    def __init__(self, num_class=56):
        super(BaseModel, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

        self.learning_rate = 1e-3

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # classifier
        self.dense = nn.Linear(256, num_class)  # 배치 8 기준 640?
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = x[:, :, :, :512]
        x = x.unsqueeze(1)
        # init bn
        x = self.bn_init(x)
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
        # classifier
        x = x.view(x.size(0), -1)
        # print("Lin input", x.shape)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))
        # print(x.shape)

        return logit

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, y)
        self.train_total_loss.append(train_loss)
        return train_loss

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        valid_loss = F.cross_entropy(logits, y)
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())
        self.roc_aucs.append(rocauc)
        self.valid_total_loss.append(valid_loss)
        self.log("val_loss", torch.tensor([valid_loss]))

        return valid_loss

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        test_loss = F.cross_entropy(logits, y)
        # test roc auc 출력 위함
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())

        self.roc_aucs.append(rocauc)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []


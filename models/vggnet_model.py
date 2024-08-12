from models.resnet_arch import ResNet, BasicBlock, Bottleneck

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

roc_aucs=[]

# 모델
class VggNet(pl.LightningModule):
    def __init__(self):
        super(VggNet, self).__init__()
        self.num_tags = 8
        self.learning_rate = 1e-3
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2),  # (in_channels, out_channels, kernel_size, stride, padding)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.mp2x2_dropout = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.conv7b = nn.Sequential(
            nn.Conv2d(384, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_ml = nn.Linear(256, 56)  # 7
        self.fc_emo = nn.Linear(256, self.num_tags)  # 8

    def forward(self, x):
        # 256 * 96 * 1

        x = x.squeeze(dim=0)
        x = x.unsqueeze(dim=1)

        x = self.conv1(x)  # [129, 49, 64]
        x = self.conv2(x)  # [130, 50, 64]
        x = self.mp2x2_dropout(x)  # [65, 25, 64]
        x = self.conv3(x)  # [66, 26, 128]
        x = self.conv4(x)  # [67, 27, 128]
        x = self.mp2x2_dropout(x)  # [33, 13, 128]
        x = self.conv5(x)  # [34, 14, 256]
        x = self.conv6(x)  # [35, 15, 256]
        x = self.conv7(x)  # [36, 16, 384]
        x = self.conv7b(x)  # [37, 17, 512]
        x = self.conv11(x)  # [38, 18, 256]
        x = x.view(x.size(0), -1)
        ml = self.fc_ml(x)
        emo = self.fc_emo(x)

        return ml

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def get_auc(self, prd_array, gt_array):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)

        roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
        pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

        for i in range(self.num_class):
            print('%s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
        return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all

    def my_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def forward_full_song(self, x):
        yy = []
        for i in range(2):
            yy.append(self.forward(x[:, i, :][1]))
        return torch.stack(yy).mean(dim=0)

    def foward_step(self, x):
        # x = x.unsqueeze(dim=1)    # 앙상블 생략
        x = self.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.foward_step(x)

        y = y.float()
        y_hat = y_hat.float()
        loss = self.my_loss(y_hat, y)
        self.train_total_loss.append(loss)

        return {'loss': loss}

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # print("x", x)
        # print("y", y)
        y_hat = self.foward_step(x)
        y = y.float()
        y_hat = y_hat.float()
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        loss = self.my_loss(y_hat, y)
        self.roc_aucs.append(rocauc)
        self.valid_total_loss.append(loss)
        self.log("val_loss", torch.tensor([loss]))

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch     # images, labels
        y_hat = self.foward_step(x)
        y = y.float()
        y_hat = y_hat.float()
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        self.roc_aucs.append(rocauc)

        return {'loss': self.my_loss(y_hat, y), 'roc auc': rocauc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auc = torch.stack([torch.tensor([x['rocauc']]) for x in outputs]).mean()
        print('rocauc : %f' % avg_auc)
        return {'val_loss': avg_loss, 'rocauc': avg_auc}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auc = torch.stack([torch.tensor([x['rocauc']]) for x in outputs]).mean()
        print('rocauc : %f' % avg_auc)
        return {'val_loss': avg_loss, 'rocauc': avg_auc}

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []


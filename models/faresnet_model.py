import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from models.resnet_arch import BasicBlock

roc_aucs = []

class FaResNet(pl.LightningModule):
    def __init__(self, block=BasicBlock, num_classes=56):
        super(FaResNet, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

        self.block = block
        self.inplanes = 64
        self.dilaion = 1
        base_channels = 128
        depth = 26
        self.num_class = 56
        self.learning_rate = 0.001
        n_channels = [
            base_channels,
            base_channels * 2 * self.block.expansion,    # expansion=1
            base_channels * 4 * self.block.expansion
        ]
        self.prd_array = []
        self.gt_array = []

        # self.c1
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        n_blocks_per_stage = 4

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, self.block, stride=1, maxpool=[1, 2, 4],
            k1s=[3, 3, 3, 3], k2s=[1, 3, 3, 3])
        self.dropout1 = nn.Dropout(0.5)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, self.block, stride=1, maxpool=[],
            k1s=[3, 1, 1, 1], k2s=[1, 1, 1, 1])
        self.dropout2 = nn.Dropout(0.5)
        self.stage3 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, self.block, stride=1, maxpool=[],
            k1s=[1, 1, 1, 1], k2s=[1, 1, 1, 1])
        self.dropout3 = nn.Dropout(0.5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.tag_list = self.read_taglist()
        self.roc_auc = []

    def read_taglist(self):
        fn = './models/tag_list.npy'
        npy = np.load(fn)
        return npy

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)  # torch.Size([128, 1, 96, 512])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.dropout1(x)
        x = self.stage2(x)
        x = self.dropout2(x)
        x = self.stage3(x)
        x = self.dropout3(x)

        x = self.avgpool(x)
        x = x.squeeze(2).squeeze(2)
        # print(" x : ", x.shape)
        x = self.fc(x)

        return x


    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=set(),
                    k1s=[3, 3, 3, 3, 3, 3], k2s=[3, 3, 3, 3, 3, 3]):
        stage = []
        if 0 in maxpool:
            stage.append(nn.MaxPool2d(2, 2))

        for index in range(n_blocks):
            stage.append(block(in_channels, out_channels, stride=stride))     # , k1=k1s[index], k2=k2s[index]
            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
                if index + 1 == mp_pos:
                    stage.append(nn.MaxPool2d(2, 2))

        return nn.Sequential(*stage)


    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y = y.float()
        y_hat = y_hat.float()
        loss = F.cross_entropy(y_hat, y)
        self.train_total_loss.append(loss)

        return {'loss': loss}

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y = y.float()
        y_hat = y_hat.float()
        # y = y.unsqueeze(1)
        # y_hat = y_hat.squeeze(2)
        loss = F.cross_entropy(y_hat, y)
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu())
        self.roc_aucs.append(rocauc)
        self.valid_total_loss.append(loss)
        self.log("val_loss", torch.tensor([loss]))

        # for prd in y_hat.cpu():
        #     self.prd_array.append(list(np.array(prd)))
        # for gt in y.cpu():
        #     self.gt_array.append(list(np.array(gt)))

        return {'val_loss': loss}

    # def on_validation_end(self):
    #     if len(self.prd_array) > 0:
    #         roc_auc, pr_auc, _, _ = self.get_auc(self.prd_array, self.gt_array)
    #         print(roc_auc, pr_auc)
    #     # 초기화
    #     self.prd_array = []
    #     self.gt_array = []

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y = y.float()
        y_hat = y_hat.float()

        # print(y.shape) # torch.Size([32, 56])
        # print(y_hat.shape) # torch.Size([32, 56])

        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu())
        self.roc_aucs.append(rocauc)

        for prd in y_hat.cpu():
            self.prd_array.append(list(np.array(prd)))
        for gt in y.cpu():
            self.gt_array.append(list(np.array(gt)))

        return {'loss': loss}

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []
        roc_auc, pr_auc, _, _ = self.get_auc(self.prd_array, self.gt_array)
        print(roc_auc, pr_auc)


    def get_auc(self, prd_array, gt_array):
        from sklearn import metrics
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

        roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
        pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

        for i in range(self.num_class):
            print('%s,%.4f,%.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
        return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all


class AttentionAvg(nn.Module):
    def __init__(self, in_channels, out_channels, sum_all=True):
        super(AttentionAvg, self).__init__()
        self.sum_dims = [2, 3]
        if sum_all:
            self.sum_dims = [1, 2, 3]
        self.forw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.atten = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        a1 = self.forw_conv(x)
        atten = self.atten(x)
        num = atten.size(2) * atten.size(3)
        asum = atten.sum(dim=self.sum_dims, keepdim=True) + 1e-8
        return a1 * atten * num / asum


def get_model_based_on_rho(rho, config_only=False):
    # extra receptive checking
    extra_kernal_rf = rho - 7
    model_config = {
        "arch": "cp_faresnet",
        "base_channels": 128,
        "block_type": "basic",
        "depth": 26,
        "input_shape": [
            10,
            2,
            -1,
            -1
        ],
        "multi_label": False,
        "n_classes": 10,
        "prediction_threshold": 0.4,
        "stage1": {"maxpool": [1, 2, 4],
                   "k1s": [3,
                           3 - (-extra_kernal_rf > 6) * 2,
                           3 - (-extra_kernal_rf > 4) * 2,
                           3 - (-extra_kernal_rf > 2) * 2],
                   "k2s": [1,
                           3 - (-extra_kernal_rf > 5) * 2,
                           3 - (-extra_kernal_rf > 3) * 2,
                           3 - (-extra_kernal_rf > 1) * 2]},

        "stage2": {"maxpool": [], "k1s": [3 - (-extra_kernal_rf > 0) * 2,
                                          1 + (extra_kernal_rf > 1) * 2,
                                          1 + (extra_kernal_rf > 3) * 2,
                                          1 + (extra_kernal_rf > 5) * 2],
                   "k2s": [1 + (extra_kernal_rf > 0) * 2,
                           1 + (extra_kernal_rf > 2) * 2,
                           1 + (extra_kernal_rf > 4) * 2,
                           1 + (extra_kernal_rf > 6) * 2]},
        "stage3": {"maxpool": [],
                   "k1s": [1 + (extra_kernal_rf > 7) * 2,
                           1 + (extra_kernal_rf > 9) * 2,
                           1 + (extra_kernal_rf > 11) * 2,
                           1 + (extra_kernal_rf > 13) * 2],
                   "k2s": [1 + (extra_kernal_rf > 8) * 2,
                           1 + (extra_kernal_rf > 10) * 2,
                           1 + (extra_kernal_rf > 12) * 2,
                           1 + (extra_kernal_rf > 14) * 2]},
        "block_type": "basic",
        "use_bn": True,
        "weight_init": "fixup"
    }
    if config_only:
        return model_config

    return Network(model_config)


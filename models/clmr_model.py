import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import numpy as np



class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # loss 분모 부분의 negative sample 간의 내적 합만을 가져오기 위한 마스킹 행렬
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size


        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본 - augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # sim_i_j.shape = [32]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # 64,1
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class CLMRBaseModel(pl.LightningModule):
    def __init__(self, num_class=56, batch_size=32):
        super(CLMRBaseModel, self).__init__()
        self.loss = SimCLR_Loss(batch_size, temperature=0.5)
        self.roc_aucs = []

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
        self.dense = nn.Linear(256,  num_class)
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
        loss = F.cross_entropy(y_hat, y)  # loss 함수 선언
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        # acc = FM.accuracy(logits, y)     # `task` 추가: to either be `'binary'`, `'multiclass'` or `'multilabel'` but got class
        val_loss = F.cross_entropy(logits, y)
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())
        self.roc_aucs.append(rocauc)
        self.log('val_loss', val_loss, prog_bar=True)

        return val_loss

    def on_validation_epoch_end(self):
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        # acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())
        self.roc_aucs.append(rocauc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []



class CLMR_CNN(pl.LightningModule):
    def __init__(self, num_class=56, batch_size=32):
        super(CLMR_CNN, self).__init__()
        self.loss = SimCLR_Loss(batch_size, temperature=0.5)
        self.roc_aucs = []
        self.total_loss = 0

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
        self.dense = nn.Linear(256, 100)  # 배치 8 기준 640?
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = x[:, :, :, :512]
        x = x.unsqueeze(1)
        # print(x.shape)
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
        x = self.dense(x)   # 256, 100 으로 리턴 -> 추후에 100->56 필요

        return x

    def training_step(self, batch, batch_idx):
        print(batch[0].shape)

        x0, x1 = batch
        x1 = x1.squeeze(1)
        # print(batch.shape)  # torch.Size([32, 96, 512])
        y = self(x0)
        y_hat = self(x1)
        loss = self.loss(y_hat, y)
        self.total_loss += loss.item()

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)




class CLMR_classifier(pl.LightningModule):
    def __init__(self, num_class=56, batch_size=32):
        super(CLMR_classifier, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []
        self.CNN = CLMR_CNN()
        # pre-trained model 위치
        self.CNN.load_from_checkpoint('trained/clmr/cnn/lightning_logs/version_40/checkpoints/epoch=199-step=77600.ckpt')
        # classifier
        self.dense = nn.Linear(100,  num_class)

    def forward(self, x):
        # print(x.shape)
        x = self.CNN(x)
        logit = nn.Sigmoid()(self.dense(x))
        return logit

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)  # loss 함수 선언
        self.train_total_loss.append(loss)
        return loss

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        # acc = FM.accuracy(logits, y)     # `task` 추가: to either be `'binary'`, `'multiclass'` or `'multilabel'` but got class
        loss = F.cross_entropy(logits, y)
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())
        self.roc_aucs.append(rocauc)
        self.valid_total_loss.append(loss)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        rocauc = roc_auc_score(y.t().cpu(), logits.t().cpu())
        self.roc_aucs.append(rocauc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []


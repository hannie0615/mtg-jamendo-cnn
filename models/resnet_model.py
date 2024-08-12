# 라이브러리 임포트
from models.resnet_arch import ResNet, BasicBlock, Bottleneck

from scipy.stats import pearsonr
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC

auroc = AUROC(task="binary", num_classes=56)

roc_aucs = []
class ResNet34(pl.LightningModule):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

        blocktype = BasicBlock
        layers = [2, 2, 2, 2]

        self.model = nn.Sequential(
            ResNet(blocktype, layers, num_classes=56),
            nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-3)]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_f = y.float()
        y_hat_f = y_hat.float()
        loss = F.cross_entropy(y_hat_f, y_f)    # self.loss() 대신
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
        loss = F.cross_entropy(y_hat, y)
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu())
        self.roc_aucs.append(rocauc)
        self.valid_total_loss.append(loss)
        self.log("val_loss", torch.tensor([loss]))

        return {"val_loss": loss}

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
        y_hat_probs = F.softmax(y_hat.float(), dim=1)

        rocauc = roc_auc_score(y.t().cpu(), y_hat_probs.t().cpu())
        self.roc_aucs.append(rocauc)
        # print('rocauc : %f' % rocauc)

        return {'loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = []
        y_hat = []

        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)
        roc = auroc(y_hat, y)
        # print("val_auroc : ", roc)

        metrics = self._compute_metrics(y, y_hat, ['loss', 'prauc', 'rocauc'])
        metrics['val_loss'] = avg_loss

        return metrics

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = []
        y_hat = []

        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, ['loss', 'prauc', 'rocauc'])
        metrics['test_loss'] = avg_loss
        roc = auroc(y_hat, y)
        # print("test_auroc : ", roc)

        return metrics

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []


    def _compute_metrics(self, y, y_hat, metrics_list):
        metrics_res = {}
        for metric in metrics_list:
            Y, Y_hat = y, y_hat
            if metric in ['rocauc-macro', 'rocauc']:
                metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='macro')
            if metric == 'rocauc-micro':
                metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='micro')
            if metric in ['prauc-macro', 'prauc']:
                metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='macro')
            if metric == 'prauc-micro':
                metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='micro')
            if metric == 'corr_avg':
                corr, pval = [], []
                for i in range(7):
                    c, p = pearsonr(Y[:, i], Y_hat[:, i])
                    corr.append(c)
                    pval.append(p)
                metrics_res['corr_avg'] = np.mean(corr)
                metrics_res['pval_avg'] = np.mean(pval)
            if metric == 'corr':
                corr, pval = [], []
                for i in range(7):
                    c, p = pearsonr(Y[:, i], Y_hat[:, i])
                    corr.append(c)

                    pval.append(p)
                metrics_res['corr'] = corr
                metrics_res['pval'] = pval

        return metrics_res

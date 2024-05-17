import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

roc_aucs=[]

# FaResNet+ResNet34+VggNet+ CRNN
class MyEnsemble4(pl.LightningModule):
    def __init__(self, modelA, modelB, modelC, modelD):
        super(MyEnsemble4, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelA.freeze()
        self.modelB.freeze()
        self.modelC.freeze()
        self.modelD.freeze()
        self.classifier = torch.nn.Linear(224, 56)

        # self.save_hyperparameters() # Uncomment to show error

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

    def forward(self, x):
        x = x.squeeze(dim=0)
        # x = x.unsqueeze(dim=1)

        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x4 = self.modelD(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # x = x.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(self.forward(x), y)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        return F.binary_cross_entropy_with_logits(self.forward(x), y)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        roc_aucs.append(rocauc)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(roc_aucs) / len(roc_aucs))
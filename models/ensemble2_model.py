import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

roc_aucs=[]

# VGGNet + CRNN
class MyEnsemble2(pl.LightningModule):
    def __init__(self, modelA, modelB):
        super(MyEnsemble2, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelA.freeze()
        self.modelB.freeze()
        self.classifier = torch.nn.Linear(112, 56)

        # self.save_hyperparameters() # Uncomment to show error

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

    def forward(self, x):
        x = x.squeeze(dim=0)
        # x = x.unsqueeze(dim=1)
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
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
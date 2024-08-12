import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from models.vggnet_model import VggNet
from models.crnn_model import CRNN_esb

roc_aucs=[]

# VGGNet + CRNN
class MyEnsemble2(pl.LightningModule):
    def __init__(self, modelA=VggNet(), modelB=CRNN_esb()):
        super(MyEnsemble2, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

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
        loss = F.cross_entropy(self.forward(x), y)
        self.train_total_loss.append(loss)
        return loss

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = F.cross_entropy(self.forward(x), y)
        rocauc = roc_auc_score(y.t().cpu(), self.forward(x).t().cpu())
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
        x, y, _ = batch
        y_hat = self.forward(x)
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        self.roc_aucs.append(rocauc)
        return F.cross_entropy(y_hat, y)


    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []
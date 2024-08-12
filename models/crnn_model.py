# 라이브러리 임포트
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from models.clmr_model import CLMR_CNN

roc_aucs=[]
# single 훈련할 때 사용하는 CRNN
class CRNN(pl.LightningModule):
    def __init__(self, num_class=56):
        super(CRNN, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []
        self.prd_array = []
        self.gt_array = []

        self.gru_hidden_size = 32
        self.gru_num_layers = 2
        self.drop_prob = 0.2
        self.learning_rate = 0.001     # 0.0005
        self.validation_metrics = ['rocauc', 'prauc']
        self.test_metrics = ['rocauc', 'prauc']

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
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # recurrent layer
        self.gru1 = nn.GRU(input_size=32,
                           hidden_size=self.gru_hidden_size,
                           num_layers=self.gru_num_layers
                           )

        # classifier
        self.dense = nn.Linear(self.gru_hidden_size, num_class)
        self.dropout = nn.Dropout(self.drop_prob)

        # --- BasePtl ----

    def forward_full_song(self, batch):
            def cnn_forward(x):
                # init bn
                x = self.bn_init(x)
                # layer 1
                x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
                # layer 2
                x = nn.ELU()(self.bn_2(self.conv_2(x)))
                # x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
                # layer 3
                x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
                # layer 4
                x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
                # layer 5
                x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))
                # classifier
                x = x.view(-1, x.size(0), 32)

                return x

            def rnn_forward(x):
                x = x.squeeze()
                x = self.gru1(x)[0][-1]  # TODO: Check if this is correct
                x = self.dropout(x)
                logit = nn.Sigmoid()(self.dense(x))
                return logit

            def extract_features(song_idx, song_length):
                # print(song_idx, song_length)
                song_length = 2560
                song_feats = []
                for l in range(song_length // self.input_size + 1):
                    data = h5data[song_idx + l * self.input_size:song_idx + min(song_length,
                                                                                (l + 1) * self.input_size)].transpose()
                    if data.shape[1] < self.input_size * 0.25:
                        continue
                    data = np.pad(data, ((0, 0), (0, self.input_size - data.shape[1])), mode='wrap')
                    try:
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cuda'))))
                    except AssertionError:
                        # print(song_idx, song_length)
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cpu'))))

                # print("song feats", song_feats.__len__(), song_feats[0].shape)
                return torch.cat(song_feats)

            h5data, idx_list, x_lengths_list, labels_list = batch
            sequences = []
            for n, ind in enumerate(idx_list):
                sequences.append(extract_features(ind, x_lengths_list[n]))

            # print("sequences", sequences.__len__(), sequences[0].shape)
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)

            result = rnn_forward(sequences_padded)
            return result


    def forward(self, batch):
            # print("batch", batch)

            x = batch[0]  # xs(*), xlens, labels => audio(*), tags, dict
            # x, _, _ = batch
            # init bn

            x = x.unsqueeze(1)    # 앙상블 시 주석 처리
            x = self.bn_init(x)     # 8*96*512

            # layer 1
            x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

            # layer 2
            x = nn.ELU()(self.bn_2(self.conv_2(x)))
            # x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

            # layer 3
            x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

            # layer 4
            x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

            # layer 5
            x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

            # classifier
            x = x.view(-1, x.size(0), 32)

            # output, hidden = self.gru(x_pack)
            # x = self.gru1(x)[0][-1] # TODO: Check if this is correct
            x = self.gru1(x)[1][1]  # TODO: Check if this is correct

            x = self.dropout(x)
            logit = nn.Sigmoid()(self.dense(x))

            return logit

    def training_step(self, data_batch, batch_i):
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()
        loss = F.cross_entropy(y_hat, y)
        self.train_total_loss.append(loss)

        return {'loss': loss}   # self.loss(y_hat, y),

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []


    def validation_step(self, data_batch, batch_i):
        y = data_batch[1]   # ('51/1053051.mp3', ...

        y_hat = self.forward(data_batch)    # tensor
        # y = torch.stack(y).float()   # 예외 처리 생략
        y = y.float()
        y_hat = y_hat.float()

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

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []

    def test_step(self, data_batch, batch_i):
        # y = data_batch[-1]
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()
        # metrics = self._compute_metrics(y.t().cpu(), y_hat.t().cpu(), self.test_metrics)
        # roc_aucs.append(metrics['rocauc'])

        loss = F.cross_entropy(y_hat, y)
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
        # roc_auc, pr_auc, _, _ = self.get_auc(self.prd_array, self.gt_array)
        # print(roc_auc, pr_auc)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.validation_metrics)
        # print("ROC AUC : ", metrics['rocauc'])
        # print("Hi, Pycharm")      # 출력이 안되는 이슈
        # print(metrics)

        metrics['val_loss'] = avg_loss


        return metrics

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.test_metrics)
        metrics['test_loss'] = avg_loss

        return metrics


    def _compute_metrics(self, y, y_hat, metrics_list):
        metrics_res = {}
        for metric in metrics_list:
            from scipy.stats import pearsonr
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

    def get_auc(self, prd_array, gt_array):
        from sklearn import metrics
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)

        roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
        pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

        for i in range(self.num_class):
            print('%s,%.4f,%.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
        return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    # 수정하기
    def _load_model(self, load_path, map_location=None, on_gpu=True):
        last_epoch = -1
        last_ckpt_name = None

        import re
        checkpoints = os.listdir(load_path)
        for name in checkpoints:
            # ignore hpc ckpts
            if 'hpc_' in name:
                continue

            # load ckpts
            if '.ckpt' in name:
                epoch = name.split('epoch_')[1]
                epoch = int(re.sub('[^0-9]', '', epoch))

                if epoch > last_epoch:
                    last_epoch = epoch
                    last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(load_path, last_ckpt_name)
            if on_gpu:
                if map_location is not None:
                    checkpoint = torch.load(last_ckpt_path, map_location=map_location)
                else:
                    checkpoint = torch.load(last_ckpt_path)
            else:
                checkpoint = torch.load(last_ckpt_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'])

# 앙상블 할 때 사용하는 CRNN
class CRNN_esb(pl.LightningModule):
    def __init__(self, num_class=56):
        super(CRNN_esb, self).__init__()
        self.gru_hidden_size = 32
        self.gru_num_layers = 2
        self.drop_prob = 0.2
        self.learning_rate = 0.0001     # 0.0005
        self.validation_metrics = ['rocauc', 'prauc']
        self.test_metrics = ['rocauc', 'prauc']

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
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # recurrent layer
        self.gru1 = nn.GRU(input_size=32,
                           hidden_size=self.gru_hidden_size,
                           num_layers=self.gru_num_layers
                           )

        # classifier
        self.dense = nn.Linear(self.gru_hidden_size, num_class)
        self.dropout = nn.Dropout(self.drop_prob)

        # --- BasePtl ----

    def forward_full_song(self, batch):
            def cnn_forward(x):
                # init bn
                x = self.bn_init(x)
                # layer 1
                x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
                # layer 2
                x = nn.ELU()(self.bn_2(self.conv_2(x)))
                # x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
                # layer 3
                x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
                # layer 4
                x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
                # layer 5
                x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))
                # classifier
                x = x.view(-1, x.size(0), 32)

                return x

            def rnn_forward(x):
                x = x.squeeze()
                x = self.gru1(x)[0][-1]  # TODO: Check if this is correct
                x = self.dropout(x)
                logit = nn.Sigmoid()(self.dense(x))
                return logit

            def extract_features(song_idx, song_length):
                # print(song_idx, song_length)
                song_length = 2560
                song_feats = []
                for l in range(song_length // self.input_size + 1):
                    data = h5data[song_idx + l * self.input_size:song_idx + min(song_length,
                                                                                (l + 1) * self.input_size)].transpose()
                    if data.shape[1] < self.input_size * 0.25:
                        continue
                    data = np.pad(data, ((0, 0), (0, self.input_size - data.shape[1])), mode='wrap')
                    try:
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cuda'))))
                    except AssertionError:
                        # print(song_idx, song_length)
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cpu'))))

                # print("song feats", song_feats.__len__(), song_feats[0].shape)
                return torch.cat(song_feats)

            h5data, idx_list, x_lengths_list, labels_list = batch
            sequences = []
            for n, ind in enumerate(idx_list):
                sequences.append(extract_features(ind, x_lengths_list[n]))

            # print("sequences", sequences.__len__(), sequences[0].shape)
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)

            result = rnn_forward(sequences_padded)
            return result


    def forward(self, batch):
            # print("batch", batch)

            x = batch  # xs(*), xlens, labels => audio(*), tags, dict
            # x, _, _ = batch
            # init bn

            x = x.unsqueeze(1)    # 앙상블 시 주석 처리
            x = self.bn_init(x)     # 8*96*512

            # layer 1
            x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

            # layer 2
            x = nn.ELU()(self.bn_2(self.conv_2(x)))
            # x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

            # layer 3
            x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

            # layer 4
            x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

            # layer 5
            x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

            # classifier
            x = x.view(-1, x.size(0), 32)

            # output, hidden = self.gru(x_pack)
            # x = self.gru1(x)[0][-1] # TODO: Check if this is correct
            x = self.gru1(x)[1][1]  # TODO: Check if this is correct

            x = self.dropout(x)
            logit = nn.Sigmoid()(self.dense(x))

            return logit

    def training_step(self, data_batch, batch_i):
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()

        return {'loss': F.binary_cross_entropy(y_hat, y)}   # self.loss(y_hat, y),


    def validation_step(self, data_batch, batch_i):
        y = data_batch[1]   # ('51/1053051.mp3', ...

        y_hat = self.forward(data_batch)    # tensor
        # y = torch.stack(y).float()   # 예외 처리 생략
        y = y.float()
        y_hat = y_hat.float()

        metrics = self._compute_metrics(y.t().cpu(), y_hat.t().cpu(), self.validation_metrics)
        # print(" ROC AUC : ", metrics['rocauc'], end='')

        return {
            'val_loss': F.binary_cross_entropy(y_hat, y),  # self.loss(y_hat, y),
            'y': y.cpu().numpy(),
            'y_hat': y_hat.cpu().numpy()
        }

    def test_step(self, data_batch, batch_i):
        # y = data_batch[-1]
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()
        # metrics = self._compute_metrics(y.t().cpu(), y_hat.t().cpu(), self.test_metrics)
        # roc_aucs.append(metrics['rocauc'])

        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        roc_aucs.append(rocauc)
        return {
            'test_loss': F.binary_cross_entropy(y_hat, y),  # self.loss(y_hat, y),
            'y': y.cpu().numpy(),
            'y_hat': y_hat.cpu().numpy()
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.validation_metrics)
        # print("ROC AUC : ", metrics['rocauc'])
        # print("Hi, Pycharm")      # 출력이 안되는 이슈
        # print(metrics)

        metrics['val_loss'] = avg_loss


        return metrics

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.test_metrics)
        metrics['test_loss'] = avg_loss

        return metrics

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(roc_aucs) / len(roc_aucs))

    def _compute_metrics(self, y, y_hat, metrics_list):
        metrics_res = {}
        for metric in metrics_list:
            from scipy.stats import pearsonr
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

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    # 수정하기
    def _load_model(self, load_path, map_location=None, on_gpu=True):
        last_epoch = -1
        last_ckpt_name = None

        import re
        checkpoints = os.listdir(load_path)
        for name in checkpoints:
            # ignore hpc ckpts
            if 'hpc_' in name:
                continue

            # load ckpts
            if '.ckpt' in name:
                epoch = name.split('epoch_')[1]
                epoch = int(re.sub('[^0-9]', '', epoch))

                if epoch > last_epoch:
                    last_epoch = epoch
                    last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(load_path, last_ckpt_name)
            if on_gpu:
                if map_location is not None:
                    checkpoint = torch.load(last_ckpt_path, map_location=map_location)
                else:
                    checkpoint = torch.load(last_ckpt_path)
            else:
                checkpoint = torch.load(last_ckpt_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'])


class CRNN_clmr(pl.LightningModule):
    def __init__(self, num_class=56):
        super(CRNN_clmr, self).__init__()
        self.roc_aucs = []
        self.train_loss = []
        self.valid_loss = []
        self.valid_total_loss = []
        self.train_total_loss = []

        # pre-trained model 위치
        self.CNN = CLMR_CNN()
        self.CNN.load_from_checkpoint(
            'trained/clmr/cnn/lightning_logs/version_40/checkpoints/epoch=199-step=77600.ckpt')
            # 'trained/clmr/new_cnn/lightning_logs/version_0/checkpoints/epoch=199-step=83600.ckpt')

        self.gru_hidden_size = 32
        self.gru_num_layers = 2
        self.drop_prob = 0.2
        self.learning_rate = 0.0001     # 0.0005
        self.validation_metrics = ['rocauc', 'prauc']
        self.test_metrics = ['rocauc', 'prauc']

        # recurrent layer
        self.gru1 = nn.GRU(input_size=100,
                           hidden_size=self.gru_hidden_size,
                           num_layers=self.gru_num_layers
                           )

        # classifier
        self.dense = nn.Linear(self.gru_hidden_size, num_class)
        self.dropout = nn.Dropout(self.drop_prob)

        # --- BasePtl ----

    def forward_full_song(self, batch):
            def cnn_forward(x):
                # init bn
                x = self.CNN(x)
                # classifier
                x = x.view(-1, x.size(0), 100)

                return x

            def rnn_forward(x):
                x = x.squeeze()
                x = self.gru1(x)[0][-1]  # TODO: Check if this is correct
                x = self.dropout(x)
                logit = nn.Sigmoid()(self.dense(x))
                return logit

            def extract_features(song_idx, song_length):
                # print(song_idx, song_length)
                song_length = 2560
                song_feats = []
                for l in range(song_length // self.input_size + 1):
                    data = h5data[song_idx + l * self.input_size:song_idx + min(song_length,
                                                                                (l + 1) * self.input_size)].transpose()
                    if data.shape[1] < self.input_size * 0.25:
                        continue
                    data = np.pad(data, ((0, 0), (0, self.input_size - data.shape[1])), mode='wrap')
                    try:
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cuda'))))
                    except AssertionError:
                        # print(song_idx, song_length)
                        song_feats.append(cnn_forward(torch.tensor([[data]], device=torch.device('cpu'))))

                # print("song feats", song_feats.__len__(), song_feats[0].shape)
                return torch.cat(song_feats)

            h5data, idx_list, x_lengths_list, labels_list = batch
            sequences = []
            for n, ind in enumerate(idx_list):
                sequences.append(extract_features(ind, x_lengths_list[n]))

            # print("sequences", sequences.__len__(), sequences[0].shape)
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)

            result = rnn_forward(sequences_padded)
            return result


    def forward(self, batch):
            # print("batch", batch)

            x = batch[0]  # xs(*), xlens, labels => audio(*), tags, dict
            # x, _, _ = batch
            # init bn

            # x = x.unsqueeze(1)    # 앙상블 시 주석 처리
            x = self.CNN(x)

            # classifier
            x = x.view(-1, x.size(0), 100)

            # output, hidden = self.gru(x_pack)
            # x = self.gru1(x)[0][-1] # TODO: Check if this is correct
            x = self.gru1(x)[1][1]  # TODO: Check if this is correct

            x = self.dropout(x)
            logit = nn.Sigmoid()(self.dense(x))

            return logit

    def training_step(self, data_batch, batch_i):
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()
        loss = F.cross_entropy(y_hat, y)
        self.train_total_loss.append(loss)

        return {'loss': loss}   # self.loss(y_hat, y),

    def on_train_epoch_end(self):
        print("\n train loss : ", sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_loss.append(sum(self.train_total_loss) / len(self.train_total_loss))
        self.train_total_loss = []

    def validation_step(self, data_batch, batch_i):
        y = data_batch[1]   # ('51/1053051.mp3', ...

        y_hat = self.forward(data_batch)    # tensor
        # y = torch.stack(y).float()   # 예외 처리 생략
        y = y.float()
        y_hat = y_hat.float()

        loss = F.binary_cross_entropy(y_hat, y)
        self.valid_total_loss.append(loss)
        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu())
        self.roc_aucs.append(rocauc)

        return {
            'val_loss': loss,  # self.loss(y_hat, y),
            'y': y.cpu().numpy(),
            'y_hat': y_hat.cpu().numpy()
        }

    def on_validation_epoch_end(self):
        print("\n valid loss : ", sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_loss.append(sum(self.valid_total_loss) / len(self.valid_total_loss))
        self.valid_total_loss = []
        print("\n valid roc auc : ", sum(self.roc_aucs) / len(self.roc_aucs))
        self.roc_aucs = []

    def test_step(self, data_batch, batch_i):
        # y = data_batch[-1]
        y = data_batch[1]
        y_hat = self.forward(data_batch)
        try:
            y = torch.stack(y).float()
        except:
            y = y.float()
        y_hat = y_hat.float()
        # metrics = self._compute_metrics(y.t().cpu(), y_hat.t().cpu(), self.test_metrics)
        # roc_aucs.append(metrics['rocauc'])

        rocauc = roc_auc_score(y.t().cpu(), y_hat.t().cpu(), average='macro')
        roc_aucs.append(rocauc)
        return {
            'test_loss': F.binary_cross_entropy(y_hat, y),  # self.loss(y_hat, y),
            'y': y.cpu().numpy(),
            'y_hat': y_hat.cpu().numpy()
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.validation_metrics)
        # print("ROC AUC : ", metrics['rocauc'])
        # print("Hi, Pycharm")      # 출력이 안되는 이슈
        # print(metrics)

        metrics['val_loss'] = avg_loss


        return metrics

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = []
        y_hat = []
        for output in outputs:
            y.append(output['y'])
            y_hat.append(output['y_hat'])

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        metrics = self._compute_metrics(y, y_hat, self.test_metrics)
        metrics['test_loss'] = avg_loss

        return metrics

    def on_test_epoch_end(self):
        print("\n test roc auc : ", sum(roc_aucs) / len(roc_aucs))
        self.loss_plot(self.train_loss, self.valid_loss)

    def _compute_metrics(self, y, y_hat, metrics_list):
        metrics_res = {}
        for metric in metrics_list:
            from scipy.stats import pearsonr
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

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    # 수정하기
    def _load_model(self, load_path, map_location=None, on_gpu=True):
        last_epoch = -1
        last_ckpt_name = None

        import re
        checkpoints = os.listdir(load_path)
        for name in checkpoints:
            # ignore hpc ckpts
            if 'hpc_' in name:
                continue

            # load ckpts
            if '.ckpt' in name:
                epoch = name.split('epoch_')[1]
                epoch = int(re.sub('[^0-9]', '', epoch))

                if epoch > last_epoch:
                    last_epoch = epoch
                    last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(load_path, last_ckpt_name)
            if on_gpu:
                if map_location is not None:
                    checkpoint = torch.load(last_ckpt_path, map_location=map_location)
                else:
                    checkpoint = torch.load(last_ckpt_path)
            else:
                checkpoint = torch.load(last_ckpt_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'])

    def loss_plot(self, list1, list2):
        import matplotlib.pyplot as plt
        plt.plot(list1, label=str(list1)[:5])
        plt.plot(list2, label=str(list2)[:5])
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training Loss over Epochs')
        plt.show()

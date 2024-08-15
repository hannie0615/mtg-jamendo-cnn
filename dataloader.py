# 필요한 라이브러리 임포트
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import torchaudio.transforms as T
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
from sklearn import preprocessing
def data_clmr_load(root):
    train_data = []

    dcase = os.listdir(root)
    for d in dcase:
        fn = os.path.join(root, d)
        audio = np.array(np.load(fn))
        if audio.shape[1] < 512:
            continue
        slice = audio[:, 0:512]
        slice = scaler.fit_transform(slice)
        slice2, _, _ = np.array(augmentations(audio))
        slice2 = slice2[:, 0:512]
        slice2 = scaler.fit_transform(slice2)
        train_data.append([slice.astype('float32'), slice2.astype('float32')])

    return train_data


# train, validation, test 데이터셋 만들기
def data_load(root='./data', tag='./tags', annotation=False):
    train_data = []
    val_data = []
    test_data = []

    subset = 'moodtheme'
    split = 0

    for mode in ['train', 'validation', 'test']:
        fn = tag + '/split-%d/%s_%s_dict.pickle' % (split, subset, mode)
        with open(fn, 'rb') as pf:
            dict = pickle.load(pf)
        print(fn)
        for idx in range(len(dict)):
            if idx % 100 == 0:
                print(str(idx)+'/'+str(len(dict)))

            tags = dict[idx]['tags']
            fn = os.path.join(root, dict[idx]['path'][:-3] + 'npy')

            audio = np.array(np.load(fn))
            length = 0  # length = int(audio.shape[1] / 2)

            if mode == 'train':
                length = 0
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    if annotation:
                        slices = augmentations(slice)
                        for j in range(3):
                            train_data.append([slices[j].astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    elif annotation is False:
                        train_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    length += 512

                # train_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])

            elif mode == 'validation':
                length = 0
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    if annotation:
                        slices = augmentations(slice)
                        for j in range(3):
                            val_data.append([slices[j].astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    elif annotation is False:
                        val_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])

                    length += 512
                # val_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])

            else:
                length = 0
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    test_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])

                    length += 512
                # test_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])
    return train_data, val_data, test_data



# 증강 함수 정의
def augmentations(mel_spectrogram):
    mel_1 = mel_spectrogram
    mel_2 = mel_spectrogram
    mel_3 = mel_spectrogram

    # 주파수 마스킹
    freq_mask_param = 15
    num_mel_channels = mel_1.shape[0]
    f0 = np.random.uniform(low=0.0, high=num_mel_channels)
    f = int(np.clip(f0, 0, num_mel_channels - freq_mask_param))
    mel_1[f:f + freq_mask_param, :] = 0

    # 시간 마스킹
    time_mask_param = 35
    num_time_steps = mel_2.shape[1]
    t0 = np.random.uniform(low=0.0, high=num_time_steps)
    t = int(np.clip(t0, 0, num_time_steps - time_mask_param))
    mel_2[:, t:t + time_mask_param] = 0

    # 노이즈 추가
    noise = np.random.randn(*mel_3.shape)
    mel_3 += noise * 0.005

    return mel_1, mel_2, mel_3

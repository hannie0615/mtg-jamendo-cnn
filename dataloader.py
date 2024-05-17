# 필요한 라이브러리 임포트
import pickle
import os
import numpy as np

# train, validation, test 데이터셋 만들기
def data_load(root='./data', tag='./tags'):

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
            length = int(audio.shape[1] / 2)

            if mode == 'train':
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    train_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    length += 512
                # train_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])
            elif mode == 'validation':
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    val_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    length += 512
                # val_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])
            else:
                for i in range(3):
                    slice = audio[:, length:length + 512]
                    if slice.shape[1] != 512:  # 꼭 필요
                        break
                    test_data.append([slice.astype('float32'), tags.astype('float32'), dict[idx]['path']])
                    length += 512
                # test_data.append([audio.astype('float32'), tags.astype('float32'), dict[idx]['path']])
    return train_data, val_data, test_data


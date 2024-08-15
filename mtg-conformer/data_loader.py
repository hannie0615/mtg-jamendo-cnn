import random
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class AudioFolder(Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = './split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)  # 경로
        self.get_dictionary(fn)

    def __getitem__(self, index):
        # 경로
        fn = os.path.join(self.root, self.dictionary[index]['path'][:-3]+'npy')
        audio = np.array(np.load(fn))
        audio = audio[:,:512]
        tags = self.dictionary[index]['tags']
        return audio.astype('float32'), tags.astype('float32'), self.dictionary[index]['path']

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)



def get_audio_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=0):
    data_loader = DataLoader(dataset=AudioFolder(root, subset, tr_val, split),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader

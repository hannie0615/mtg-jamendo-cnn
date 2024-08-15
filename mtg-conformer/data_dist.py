import librosa
import numpy as np
import pickle
from matplotlib import pyplot as plt


def get_dictionary(fn):
    with open(fn, 'rb') as pf:
        dictionary = pickle.load(pf)
    return dictionary


def show_distribution():
    fn = 'split-0/moodtheme_train_dict.pickle'
    dict = get_dictionary(fn)

    #  9910: {'path': '81/1420681.mp3', 'duration': 13091.0, 'tags': array([0., ..])
    print(len(dict)) # 9949
    dur = []
    cnt = 0
    for i in range(len(dict)):
        if dict[i]['duration'] < 15000:
            dur.append(dict[i]['duration'])
            cnt += 1

    dur = np.array(dur)
    print(dur.min(), dur.max())
    print(cnt/len(dict))    # duration<15000 의 비율은 전체의 0.63

    plt.hist(dur, bins=10)
    plt.show()

# duration<15000 의 비율은 전체의 63% 이다.



if __name__=='__main__':
    fn = 'C:/Users/KETI/Mtg-jamendo-dataset/data/splits/split-0/moodtheme_train_dict.pickle'
    with open(fn, 'rb') as pf:
        dictionary = pickle.load(pf)

    data = dictionary[0]
    print(data)
    tags = dictionary[0]['tags']
    print(tags)
    print(len(tags))

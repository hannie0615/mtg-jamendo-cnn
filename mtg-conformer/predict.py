import argparse
import csv

import os
import numpy as np
import torch
import time
import pydub
from matplotlib import pyplot as plt

from model import CNN512 as CNN
from shownpy import librosa_read

class Prediction(object):
    def __init__(self, input_audio, input_npy):
        self.input_audio = input_audio
        self.input_npy = np.array(np.load(input_npy))
        self.tag_list = './tags/moodtheme_split.txt'
        self.num_class = 56
        self.build_model()  # 모델 로드
        self.moodtheme_tags()   # self.tag list 가져오기

    def moodtheme_tags(self):
        # 태그
        tag_list = []
        f = open(self.tag_list, 'r')
        lines = f.readlines()
        for line in lines:
            l = line[13:-1]
            tag_list.append(l)
        self.tag_list = tag_list

    def printTags(self):
        for t in range(self.tag_list):
            print(t)

    # 플롯 출력하기
    def plot(self):
        print('plot')

    def build_model(self):
        # 모델 로드
        model = CNN(num_class=self.num_class)  # moodtheme=56
        model.cuda()
        model.load_state_dict(torch.load('./load_models/best_model.pth'))
        model.eval()
        self.model = model

    def _readMP3(self, input_audio):
        sr, y = librosa_read(input_audio)
        return y

    """
    'npy path' --(convert)--> npy array 
    """
    def _sliceNPY(self, input_npy):
        input_npy = np.array(input_npy)
        # mid = int(input_npy.shape[1] / 2)
        input_npy = input_npy[:, :512]  # [:, mid:mid+512]
        npy = input_npy.copy()
        npy.resize(1, 96, 512, refcheck=False)
        # npy.resize(1, 96, 6590, refcheck=False) # add
        npy = torch.Tensor(npy)
        npy = npy.cuda()
        return npy

    def predict(self, input):   # mp3로 받기
        start_t = time.time()

        npy = self._sliceNPY(input)
        out = self.model(npy)

        logits = out[0]
        logits = logits.detach().cpu().numpy()
        predicted = np.argmax(logits, axis=0).flatten()

        end_t = time.time()
        print(end_t - start_t)
        result = int(predicted)

        return result, self.tag_list[result]

    def predict_one(self, video_path):
        """
        video_path를 읽어와 one 'video'에 따른 prediction을 리턴한다
        :return: app, video, starttime, endtime, event, event_num
        """
        self.app = 'AED'
        video_name = video_path[-10:-4]

        starttime = '00:00'
        endtime = '05:02'

        result, event = self.predict(self._readMP3(audio))

        return self.app, video_name, starttime, endtime, event, result+1


    def showAccuracy(self, mode='npy', split=5):
        """
        data/split-0의 test.tsv에서 1000개의 샘플에 대해 model의 accuracy를 구한다.
        :param mode: 'mp3' or 'npy'
        """
        length = 1000
        path = 'C:/Users/KETI/Mtg-jamendo-dataset/data/splits/split-%d/autotagging_moodtheme-test.tsv' % split
        p = open(path, 'r')
        lines = p.readlines()
        del lines[0]

        cnt = 0
        for i in range(length):
            line = lines[i]
            answer_list = []
            temp = line.strip().split("\t")
            folder = temp[3][:2]
            file = temp[3][3:-3]

            if mode == 'mp3':
                # mp3로 입력받기
                audio = 'D:/Mtg-jamendo-dataset/audio/' + folder + '/' + file + 'mp3'
                pred = self.predict(self._readMP3(audio))

            elif mode == 'npy':
                # npy로 입력받기
                mel = 'D:/Mtg-jamendo-dataset/melspecs/' + folder + '/' + file + 'npy'
                pred = self.predict(np.load(mel))

            else:
                print("please check the Mode is 'mp3' or 'npy'")
                break

            for i in range(5, len(temp)):
                answer_list.append(temp[i][13:])

            print(answer_list)
            print(pred)

            if pred in answer_list:
                cnt += 1
                print('cnt = '+str(cnt))

        print('\nlength = %d' % (length))
        print('accuracy = %.4f' % (cnt/length))    # train.tsv - 1000개중에 139..? # test.tsv - 2000개 중에 315개



def countFile():
    fn = 'C:/Users/KETI/Mtg-jamendo-dataset/data/splits/split-5/'
    path = fn+'autotagging_moodtheme-test.tsv'
    p = open(path, 'r')
    lines = p.readlines()
    del lines[0]
    test = []
    print(len(lines))
    for line in lines:
        if line not in test:
            test.append(line)
    print(len(test))

    path2 = fn+'autotagging_moodtheme-validation.tsv'
    p2 = open(path2, 'r')
    lines2 = p2.readlines()
    del lines2[0]
    val = []
    print(len(lines2))
    for line in lines2:
        if line not in test:
            if line not in val:
                val.append(line)
    print(len(val))

    path3 = fn+'autotagging_moodtheme-train.tsv'
    p3 = open(path3, 'r')
    lines3 = p3.readlines()
    del lines3[0]

    train = []
    print(len(lines3))
    for line in lines3:
        if line not in test:
            if line not in val:
                if line not in train:
                    train.append(line)
    print(len(train))

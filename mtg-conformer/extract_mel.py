import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import soundfile as sf

# ------ plot은 꼭 필요 x -----------
def plot_spectrogram(file_name, hop_length, y_axis="linear"):
    audio, sr = librosa.load(file_name)
    plt.figure(figsize=(25, 10))
    # log-amplitude spectrogram
    log_audio = librosa.power_to_db(audio)
    librosa.display.specshow(log_audio, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()


def adding_white_noise(data, sr=22050, noise_rate=0.005):
    # noise 추가
    wn = np.random.randn(len(data))
    data_wn = data + noise_rate * wn
    # plot_time_series(data_wn)
    # librosa.output.write_wav('./white_noise.wav', data, sr=sr)  # 저장
    # print('White Noise 저장 성공')

    return data_wn


def minus_sound(data, sr=22050):
    # 위상을 뒤집는 것으로서 원래 소리와 똑같이 들린다.
    temp_numpy = (-1) * data
    plot_time_series(temp_numpy)
    librosa.output.write_wav('./minus_data.wav', temp_numpy, sr=sr)

    return data


def extract_stft_feature(file_name):
    print(f'file name: {file_name}')
    audio, sample_rate = librosa.load(file_name)
    stft_audio = librosa.stft(audio, n_fft=2048, hop_length=512)
    y_audio = np.abs(stft_audio) ** 2

    return y_audio


def extract_mfcc_feature(file_name):
    print(f'file name: {file_name}')
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=96, hop_length=512)
    mfcc = mfccs[:,:512]    # (96, 512)

    return mfcc



error_list = []

audio_path = 'D:/Mtg-jamendo-dataset/audio/'
mel_path = 'D:/Mtg-jamendo-dataset/melspecs_aug_3/'
if not os.path.exists(mel_path):
    os.mkdir(mel_path)

def run_mel():
    ## step 1
    file_list = os.listdir(audio_path)
    for f in file_list:
        mp3_list = os.listdir(audio_path+f)

        if os.path.exists(mel_path + f):    # f: [00, 01, 02, ... ]
            pass
        else:
            os.mkdir(mel_path + f)

            for mp3 in mp3_list:
                print(f'file name: {audio_path + f + '/' + mp3}')
                try:
                    audio, sample_rate = librosa.load(audio_path+f+'/'+mp3)
                    audio = adding_white_noise(audio)
                    # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=96, hop_length=512)
                    # np.save(mel_path+f+'/'+mp3[:-4]+'.npy', mfccs)
                    # 멜 스펙트로그램으로 전환
                    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=96)
                    log_mels = librosa.power_to_db(S, ref=np.max)   # ref 지정 안하면 에러

                    length = int(log_mels.shape[1] / 2)
                    mels = log_mels[:, length:length + 512*3]
                    np.save(mel_path+f+'/'+mp3[:-4]+'.npy', mels)
                except:
                    error_list.append(str(f + '/' + mp3))
    np.save('error_list.npy', error_list)


def run_error():
    ## step 2
    import shutil
    error_list = np.load('error_list.npy')
    for error in error_list:
        print(error)
        # mp3_file = audio_path+error[:-4]+'.mp3'
        to_path = mel_path+error[:-4]+'.npy'
        from_path = 'D:/Mtg-jamendo-dataset/melspecs/'+error[:-4]+'.npy'
        if os.path.isfile(to_path):
            os.remove(to_path)
        shutil.copyfile(from_path, to_path)

def run_length():
    ## step 3
    # 목표 길이 512*3
    length_list= []
    file_list = os.listdir(mel_path)
    for f in file_list:
        # f: [00, 01, 02, ... ]
        npy_list = os.listdir(mel_path + f)
        for npy in npy_list:
            np01 = np.load(mel_path + f +'/'+ npy)
            if np01.shape[1] != 512*3:
                # 512*3 크기가 아닌 것들 제외
                length_list.append(str(f + '/' + npy))
    np.save('length_list.npy', length_list)


def run_org():
    ## step 4
    length_list = np.load('length_list.npy')
    for len in length_list:
        print(len)
        path = mel_path + len
        if os.path.isfile(path):
            os.remove(path)

        audio, sample_rate = librosa.load(audio_path + len[:-4]+'.mp3')
        audio = adding_white_noise(audio)
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=96)
        log_mels = librosa.power_to_db(S, ref=np.max)  # ref 지정 안하면 에러

        length = 0
        mels = log_mels[:, length:length + 512 * 3]
        np.save(path, mels)


def run_padding():
    ## step 5
    length_list = np.load('length_list.npy')
    for len in length_list:
        path = mel_path + len
        print(path)
        np01 = np.load(path)
        if np01.shape[1] < 512*3:
            padding_shape = ((0, 0), (0, 512 * 3 - np01.shape[1]))  # 목표 길이 512*3
            padded_np = np.pad(np01, padding_shape, 'constant', constant_values=0)
            np.save(path, padded_np)
            print(padded_np.shape)
        else: pass


def annotation(y, sr):
    # 시간 스트레칭
    y_stretched = librosa.effects.time_stretch(y, rate=0.8)

    # 피치 시프트
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=4)

    # 노이즈 추가
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise

    # 타임 쉬프팅
    y_shifted_time = np.roll(y, sr)  # 1초 앞으로 이동

    # 볼륨 조절
    y_louder = y * 1.2
    y_filtered = bandpass_filter(y, lowcut=500.0, highcut=2000.0, fs=sr)

    # 리버브
    y_reverb = add_reverb(y, sr)

    # 결과 저장
    # sf.write('stretched_music_file.wav', y_stretched, sr)

    # 7개의 annotation 반환
    return y, y_stretched, y_shifted, y_noisy, y_shifted_time, y_louder, y_filtered, y_reverb


# EQ 변화 (단순한 필터링)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# 리버브 추가를 librosa와 numpy로 구현
def add_reverb(y, sr, reverb_factor=0.3):
    reverb_signal = np.convolve(y, np.ones(int(reverb_factor * sr)), mode='full')[:len(y)]
    y_reverb = y + reverb_signal * 0.2
    return y_reverb


if __name__ == '__main__':

    npy = np.load('D:/Mtg-jamendo-dataset/melspecs_aug_3/11/1315511.npy')
    print(npy.shape)    # (96,102)


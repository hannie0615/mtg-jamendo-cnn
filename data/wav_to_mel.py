import librosa
import librosa.display
import numpy as np
import os

# WAV 파일 경로 설정
dcase_dataset = 'D:/Dcase-task1/TAU-urban-acoustic-scenes-2020-mobile-development/audio'
mel_dataset = 'D:/Dcase-task1/mel'


dcase_list = os.listdir(dcase_dataset)
for d in dcase_list:
    print(d)
    wav_file = os.path.join(dcase_dataset, d)
    # WAV 파일 로드
    y, sr = librosa.load(wav_file, sr=None)

    # Mel Spectrogram 생성
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)

    # Mel Spectrogram을 Decibel 단위로 변환
    S_dB = librosa.power_to_db(S, ref=np.max)

    # .npy 파일로 저장
    output_npy_path = os.path.join(mel_dataset, str(d[:-4])+'.npy')
    np.save(output_npy_path, S_dB)


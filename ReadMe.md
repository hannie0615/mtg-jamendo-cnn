## Music Mood-theme classification (with MTG Dataset)

- 사용한 데이터셋 : MTG jamendo dataset - melspecs(.npy) 파일

### 파일 구조
```
# 파일 구조 
  ├── tags/                         # 오디오 라벨 태깅
  ├── data/                         
  |       ├── melspecs              # 오디오 데이터 - melspecs 원본 파일
  |       ├── melspecs_5            # 오디오 데이터 - melspecs 파일을 512크기로 미리 슬라이싱
  |
  ├── modles/                       # cnn 모델
  ├── trained/                      # 학습된 모델 저장(학습 시 자동생성)
  |
  ├── exp_***_lightning.py          # 학습실행 코드
  |
  ├── README.md                     # This file
  └── requirements.txt              # External module dependencies(라이브러리)
```  
  
_Q. data/melspecs_5를 사용하는 이유?_  
_A. 학습 시 데이터를 로드하는 시간 단축_

_Q. 데이터(.wav, .mel) 다운로드 받는 경로  
_A. https://github.com/MTG/mtg-jamendo-dataset


### 실행문

- 가상환경 설정
- - numpy 버전은 1.xx.xx 버전으로 설치해야 함(lightning 버전 충돌 issue) 
```commandline
pip install -r requirements.txt
```

```
# baseline 실행
> python exp_baseline_lightning.py
------
# resnet 실행
> python exp_resnet_lightning.py

# vggnet 실행
> python exp_vggnet_lightning.py

# crnn 실행
> python exp_crnn_lightning.py

# faresnet 실행
> python exp_faresnet_lightning.py
------
# ensemble1(resnet+crnn) 실행
> python exp_ensemble1_lightning.py

# ensemble2(vggnet+crnn) 실행
> python exp_ensemble2_lightning.py

# ensemble3(faresnet+crnn) 실행
> python exp_ensemble3_lightning.py

# ensemble4(faresnet+resnet+vggnet+crnn) 실행
> python exp_ensemble4_lightning.py

```


###  info
- parameter 조절
```commandline

exp_***_lightning.py 내 config에서 parameter 조절 가능

config = {
    'epochs': 200,
    'batch_size': 32,
    'root': './data/melspecs_5',    # data_path
    'tag_path': './tags',           # tag_file_path
    'model_save_path': './trained/baseline/'    
}
```

- model 기본 구조  
```pytorch-lightning.LightningModule``` 사용  
  
- optimizer  
```Adam optimizer``` 사용

- Loss  
```torch.nn.Functional.binary_cross_entropy()``` 사용
  


### 실행 결과

```
# 데이터 로드
.
.
9900/9949
.
3800/3802
.
4200/4231
.
.
# 모델 출력
  | Name       | Type     | Params
----------------------------------------
0 | modelA     | FaResNet | 3.6 M 
1 | modelB     | ResNet34 | 11.2 M
2 | modelC     | VggNet   | 4.0 M 
3 | modelD     | CRNN_esb | 459 K 
4 | classifier | Linear   | 12.6 K
----------------------------------------
12.6 K    Trainable params
19.2 M    Non-trainable params
19.2 M    Total params
76.727    Total estimated model params size (MB)
.
.
# 학습 진행
Epoch 0: 100%|██████████| 311/311 [00:13<00:00, 23.54it/s, v_num=422]
Epoch 1: 100%|██████████| 311/311 [00:13<00:00, 23.69it/s, v_num=422]
.
.
# 테스트
Testing DataLoader 0: 100%|██████████| 133/133 [00:07<00:00, 17.44it/s]
 test roc auc :  0.716668545259525
 
 

```

## Mediapipe

https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ko

구글의 오픈소스 머신러닝/인공지능 라이브러리로, 관절 추적을 활용한 실시간 손/포즈 추적 파이프라인을 제공한다.
관절의 누적된 각도를 학습하여 집합을 생성하고, 새로운 동작이 들어오면 이를 예측하여 매핑된 동작을 송출해주는 알고리즘을 이용한다.

## Model

CPU 환경에서 Visual Studio Code에서 30초씩 webcam과 Opencv를 이용하여, 총 13개의 데이터셋을 직접 수집하였고 총 13개의 sequence 파일을 생성하였다.
총 3개의 layer로 model을 training하였고, layer는 LSTM, Dense layer를 사용하였다.
Optimizer는 Adam을 사용하였고, 총 parameter의 수는 44,493으로 도출되었다. epoch는 200번으로 시행하였다.
최종 validation loss는 약 0.0164, validation accuracy는 1.0로 우수한 성능의 모델이 도출되었다.

## Structure

        D:.
|   create_dataset.py
|   test.py
|   train.ipynb
|
+---models
|       mp-13.h5
|
\---mog_dataset2
        raw_best_1716781785.npy
        raw_call_1716781785.npy
        raw_fine_1716781785.npy
        raw_happy_1716781785.npy
        raw_heart_1716781785.npy
        raw_hello_1716781785.npy
        raw_iloveyou_1716781785.npy
        raw_meet_1716781785.npy
        raw_me_1716781785.npy
        raw_peace_1716781785.npy
        raw_see_1716781785.npy
        raw_smile_1716781785.npy
        raw_what_1716781785.npy
        seq_best_1716781785.npy
        seq_call_1716781785.npy
        seq_fine_1716781785.npy
        seq_happy_1716781785.npy
        seq_heart_1716781785.npy
        seq_hello_1716781785.npy
        seq_iloveyou_1716781785.npy
        seq_meet_1716781785.npy
        seq_me_1716781785.npy
        seq_peace_1716781785.npy
        seq_see_1716781785.npy
        seq_smile_1716781785.npy
        seq_what_1716781785.npy



        

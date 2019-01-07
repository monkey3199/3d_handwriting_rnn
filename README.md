# 3d_handwriting_rnn

3D Handwriting RNN 프로젝트는 3D로 그려진 알파벳의 가속도 값으로 실제 알파벳을 분류하기 위한 RNN 모델입니다.

## Dataset
데이터는 문제가 될 소지가 있어 공유하지 않습니다.

데이터의 형식은 다음과 같습니다.

* feature: 3d로 그려지는 알파벳의 x, y, z 축으로 이뤄진 3차원 가속도 값(값의 길이는 가변)
* answer: 해당 feature의 결과값(ex. 'a')

## Model

### 1. Basic LSTM

* 3d_hw_basic.py: 단순하게 하나의 LSTM 레이어만을 이용한 모델링
* Input: (n, 154, 3)
* Test score: 3.263940857575003
* Test accuracy: 0.03834808413433818 -> 약 4%의 정확도

|Layer (type)|Output Shape|Param # |
|--|--|--|
|lstm (LSTM)|(None, 64)|17408|
|dense (Dense)|(None, 26)|1690|

### 2. CNN + LSTM

* 3d_hw_cnn+lstm.py: 가속도 별로 CNN+LSTM 모델로 만든 후 3개의 모델을 통합하는 방식을 이용한 모델링
* Input: (n, 154, 1) * 3
* Test score: 1.2917028027298176
* Test accuracy: 0.6843657846999379 -> 약 68%의 정확도

|Layer (type)|Output Shape|Param #|Connected to|
|--|--|--|--|
|input_1 (InputLayer)|(None, 154, 1)|0||
|input_2 (InputLayer)|(None, 154, 1)|0||
|input_3 (InputLayer)|(None, 154, 1)|0||
|conv1d (Conv1D)|(None, 154, 512)|2048|input_1[0][0]|
|conv1d_3 (Conv1D)|(None, 154, 512)|2048|input_2[0][0]|
|conv1d_6 (Conv1D)|(None, 154, 512)|2048|input_3[0][0]|
|batch_normalization (BatchNorma|(None, 154, 512)|2048|conv1d[0][0]|
|batch_normalization_3 (BatchNor|(None, 154, 512)|2048|onv1d_3[0][0]|
|batch_normalization_6 (BatchNor|(None, 154, 512)|2048|conv1d_6[0][0]|
|max_pooling1d (MaxPooling1D)|(None, 77, 512)|0|batch_normalization[0][0]|
|max_pooling1d_3 (MaxPooling1D)|(None, 77, 512)|0|batch_normalization_3[0][0]|
|max_pooling1d_6 (MaxPooling1D)|(None, 77, 512)|0|batch_normalization_6[0][0]|
|dropout (Dropout)|(None, 77, 512)|0|max_pooling1d[0][0]|
|dropout_3 (Dropout)|(None, 77, 512)|0|max_pooling1d_3[0][0]|
|dropout_6 (Dropout)|(None, 77, 512)|0|max_pooling1d_6[0][0]|
|conv1d_1 (Conv1D)|(None, 77, 64)|98368|dropout[0][0]|
|conv1d_4 (Conv1D)|(None, 77, 64)|98368|dropout_3[0][0]|
|conv1d_7 (Conv1D)|(None, 77, 64)|98368|dropout_6[0][0]|
|batch_normalization_1 (BatchNor|(None, 77, 64)|256|conv1d_1[0][0]|
|batch_normalization_4 (BatchNor|(None, 77, 64)|256|conv1d_4[0][0]|
|batch_normalization_7 (BatchNor|(None, 77, 64)|256|conv1d_7[0][0]|
|max_pooling1d_1 (MaxPooling1D)|(None, 38, 64)|0|batch_normalization_1[0][0]|
|max_pooling1d_4 (MaxPooling1D)|(None, 38, 64)|0|batch_normalization_4[0][0]|
|max_pooling1d_7 (MaxPooling1D)|(None, 38, 64)|0|batch_normalization_7[0][0]|
|dropout_1 (Dropout)|(None, 38, 64)|0|max_pooling1d_1[0][0]|
|dropout_4 (Dropout)|(None, 38, 64)|0|max_pooling1d_4[0][0]|
|dropout_7 (Dropout)|(None, 38, 64)|0|max_pooling1d_7[0][0]|
|conv1d_2 (Conv1D)|(None, 38, 64)|12352|dropout_1[0][0]|
|conv1d_5 (Conv1D)|(None, 38, 64)|12352|dropout_4[0][0]|
|conv1d_8 (Conv1D)|(None, 38, 64)|12352|dropout_7[0][0]|
|batch_normalization_2 (BatchNor|(None, 38, 64)|256|conv1d_2[0][0]|
|batch_normalization_5 (BatchNor|(None, 38, 64)|256|conv1d_5[0][0]|
|batch_normalization_8 (BatchNor|(None, 38, 64)|256|conv1d_8[0][0]|
|max_pooling1d_2 (MaxPooling1D)|(None, 19, 64)|0|batch_normalization_2[0][0]|
|max_pooling1d_5 (MaxPooling1D)|(None, 19, 64)|0|batch_normalization_5[0][0]|
|max_pooling1d_8 (MaxPooling1D)|(None, 19, 64)|0|batch_normalization_8[0][0]|
|lstm (LSTM)|(None, 19, 128)|98816|max_pooling1d_2[0][0]|
|lstm_3 (LSTM)|(None, 19, 128)|98816|max_pooling1d_5[0][0]|
|lstm_6 (LSTM)|(None, 19, 128)|98816|max_pooling1d_8[0][0]|
|lstm_1 (LSTM)|(None, 19, 128)|131584|lstm[0][0]|
|lstm_4 (LSTM)|(None, 19, 128)|131584|lstm_3[0][0]|
|lstm_7 (LSTM)|(None, 19, 128)|131584|lstm_6[0][0]|
|lstm_2 (LSTM)|(None, 128)|131584|lstm_1[0][0]|
|lstm_5 (LSTM)|(None, 128)|131584|lstm_4[0][0]|
|lstm_8 (LSTM)|(None, 128)|131584|lstm_7[0][0]|
|dropout_2 (Dropout)|(None, 128)|0|lstm_2[0][0]|
|dropout_5 (Dropout)|(None, 128)|0|lstm_5[0][0]|
|dropout_8 (Dropout)|(None, 128)|0|lstm_8[0][0]|
|concatenate (Concatenate)|(None, 384)|0|dropout_2[0][0] dropout_5[0][0] dropout_8[0][0]|
|dropout_9 (Dropout)|(None, 384)|0|concatenate[0][0]|
|dense (Dense)|(None, 26)|10010|dropout_9[0][0]|
|batch_normalization_9 (BatchNor|(None, 26)|104|dense[0][0]|
|activation (Activation)|(None, 26)|0|batch_normalization_9[0][0]|
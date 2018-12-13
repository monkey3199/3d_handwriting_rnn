# 3d_handwriting_rnn

3D Handwriting RNN 프로젝트는 3D로 그려진 알파벳의 가속도 값으로 결과를 분류하기 위한 RNN 모델입니다.

## data
데이터는 문제가 될 소지가 있어 공유하지 못합니다.

데이터의 형식은 다음과 같습니다.

* feature: 3d로 그려지는 알파벳의 x, y, z 축으로 이뤄진 3차원 가속도 값(값의 길이는 가변)
* answer: 해당 feature의 결과값(ex. 'a')
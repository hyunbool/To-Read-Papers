# Learning phrase representations using RNN encoder-decoder for statistical machine translation(2014)
## Introduction
* 기존의 phrase 기반의 SMT(Statistical Machine Translation) 시스템을 개선하고자 고안한 기법
    * phrase table에서 각 phrase pair에 점수를 매길 때 사용
* RNN Encoder-Decoder 모델: 두개의 RNN을 사용해 하나는 가변 길이 source sequence를 고정 길이 벡터로 맵핑하는 인코더로, 하나는 벡터를 다시 가변 길이의 타깃 시퀀스로 바꿔주는 디코더로 이용하는 기법
    * 조건부 확률을 최대화하는 방향으로 두 네트워크를 jointly하게 학습
        * joint training: 여러 개의 loss들을 더해서 최종 loss로 사용하는 방식 
* 또한 RNN Encoder-Decoder를 사용해 구문에 대한 의미론적, 구문론적 구조에 대한 정보를 저장할 수 있는 연속적인 공간 표현을 만들어 낼 수도 있다는 것을 밝힘.

## RNN Encoder-Decoder
### Preliminary: Recurrent Neural Networks
![RNN](./img/rnn.png)
* RNN(Recurrent neural network)
      * ddd

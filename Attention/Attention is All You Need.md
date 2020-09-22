# Attention Is All You Need(2017)
## Introduction
* RNN -> input과 output sequence를 구성하는 symbol의 position을 따라 계산을 수행
    * 각 시각마다 단어를 나열(align)해 h_(t-1)과 input을 이용해 h_t 시퀀스를 얻어냄
* 이런 Recurrent 모델은 병렬처리를 할 수 없기 때문에 긴 시퀀스를 다룰때나 메모리 제약이라는 단점을 가진다.
    * 시퀀스의 길이에 따른 성능 저하 문제를 해결하기 위해 Attention이 고안되었지만 대부분의 Attention 또한 Recurrent 모델을 사용하고 있다. 
* **Transformer**:
    * Recurrent를 제거해 input과 output 간 global dependancy를 구할 수 있는 기법
    * 병렬 처리도 가능!

## Model Architecture
<p align="center"><img src = "./img/39.png" width="300px" align="center"></p>
* Transformer: 기존의 인코더-디코더 모델이서 착안한 구조 가짐 + RNN을 없애고 self-attention 레이어와 Feed Forward 네트워크로만 구성

### 1. Encoder and Decoder Stacks
#### Encoder
* 6개의 레이어로 구성, 각 레이어는 두개의 sub-layer로 구성
    * multi-head self-attention layer & position-wise fully connected feed forward network
* sub-layer는 residual connection과 layer normalization으로 연결되어 있다.
    * 각 sub-layer의 output: LayerNorm(x + Sublayer(x))
* 임베딩 레이어를 포함해 모든 sub-layer는 d_model = 512차원을 가진다.

#### Decoder
* 6개의 레이어로 구성, 각 레이어는 3개의 sub-layer로 구성
    * 인코더와 동일한 두개의 sub-layer + masked multi-head attention
    <p align="center"><img src = "./img/40.gif" width="600px" align="center"></p>
    <p align="center"> 그림 참조: http://jalammar.github.io/illustrated-transformer/</p>
    
    * masking:
         * 디코더 -> 그림과 같이 항상 현재 시각 이전에 generate 된 output만 참조하기 때문에 **현재 시각 이후의 key들에 대해 masking 해줌**         
* 인코더와 마찬가지로 sub-layer들은 residual connection과 layer normalization으로 연결

### 2. Attention
<p align="center"><img src = "./img/41.gif" width="600px" align="center"></p>
<p align="center">그림 참조: https://wikidocs.net/31379</p>\

* Attention Function:
    * 주어진 query에 대해 key와의 유사도 구해 유사도를 가중치로 해 key와 맵핑되어 있는 각 value에 반영
    * 유사도가 반영된 value의 가중합 구해 리턴
    
#### 2.1 Scaled Dot-Product Attention
<p align="center"><img src = "./img/42.png" width="600px" align="center"></p>

* input은 query, key, value로 이뤄져 있다!
      * query와 key 사이의 dot product를 구함
      * 각각을 <p align="center"><img src = "./img/41.gif" width="600px" align="center"></p>로 나눠줌
      * value에 대한 weight를 얻기 위해 softmax function 적용
* 모든 과정은 단어별로 따로 수행되는 것이 아니라 일괄적으로 연산 수행
* 

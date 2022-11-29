# NLP_assignment3

# XLNet : Generalized Autoregressive Pretraining for Language Understanding

## 1. AR(Autoregressive) vs AE(Auto Encoding)
  ### AR
  - 일반적인 Language Model의 학습 방법. 이전 토큰들을 보고 다음 토큰을 예측.
  ex) ELMO, GPT
  - 방향성이 정해져야 해서 한쪽 방향의 정보만을 이용가능. 즉, 양방향 문맥을 활용하기 어려움.
  
  ### AE
  - 주어진 input에 대해 그 input을 그대로 예측.
  - Denoising AE는 noise가 섞인 input을 원래의 input으로 예측.
  ex) BERT : 주어진 input에 임의의 noise([mask])를 입히고, noise를 원래 input으로 복구.
  - [mask]를 맞추기 위해 양 방향의 정보를 이용 가능.
  - 하지만 independent assumption으로 모든 [mask]가 독립적으로 예측되므로 이들 사이의 dependency를 학습할 수 없음.
  - 또한, [mask]가 실제 fine-tuning 과정에는 등장하지 않아서, pre-training과 fine-tuning 사이에 불일치가 발생.

## 2. XLNet
  - 위의 AR과 AE 각각의 장점을 살리고 단점을 극복한 모델로 아래의 3가지 방식(2.1 ~ 2.3)을 이용해 제안됨.
  
  ### 2.1 Permutaion Language Modeling Objective
  - input sequence index의 모든 permutation을 고려한 AR 방식을 이용.
    ex) [x1, x2, x3, x4]에 대한 permutation은 4!=24개(Zt=[[1,2,3,4], [1,2,4,3], ... , [4,3,2,1]]).
    이 Zt에 대해 AR Language Model의 objective를 적용하면 아래와 같음.(mem(m)은 Transformer-XL의 memory state)
    
    - 각 토큰들은 원래 순서에 따라 positional encoding이 부여되기 때문에 토큰들의 상대적 position을 구분할 수 있음.
    - AR 방식이라 양방향 context를 고려하였고, AE 방식의 한계인 pre-training과 fine-tuning의 불일치가 없음(independent assumption이 없고 [mask]도 없기 때문).
    
  ### 2.2 Target-Aware Representation for Transformer
  

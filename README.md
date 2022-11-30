# NLP_assignment3

# XLNet : Generalized Autoregressive Pretraining for Language Understanding

## 1. AR(Autoregressive) vs AE(Auto Encoding)
  ### AR
  ![image](https://user-images.githubusercontent.com/48917098/204673978-628e0389-247e-4ed1-b987-15ba1b8f62da.png)
  - 일반적인 Language Model(LM)의 학습 방법. 이전 토큰들을 보고 다음 토큰을 예측.
  ex) ELMO, GPT
  - 방향성이 정해져야 해서 한쪽 방향의 정보만을 이용가능. 즉, 양방향 문맥을 활용하기 어려움.
  
  ### AE
  ![image](https://user-images.githubusercontent.com/48917098/204674004-c6f87321-be31-480d-918c-25ef166539c9.png)
  - 주어진 input에 대해 그 input을 그대로 예측.
  - Denoising AE는 noise가 섞인 input을 원래의 input으로 예측.
  ex) BERT : 주어진 input에 임의의 noise([mask])를 입히고, noise를 원래 input으로 복구.
  - [mask]를 맞추기 위해 양 방향의 정보를 이용 가능.
  - 하지만 independent assumption으로 모든 [mask]가 독립적으로 예측되므로 이들 사이의 dependency를 학습할 수 없음.
  - 또한, [mask]가 실제 fine-tuning 과정에는 등장하지 않아서, pre-training과 fine-tuning 사이에 불일치가 발생.

## 2. XLNet
  - 위의 AR과 AE 각각의 장점을 살리고 단점을 극복한 모델로 아래의 3가지 방식(2.1 ~ 2.3)을 이용해 제안됨.
  
  ### 2.1 Permutaion Language Modeling Objective
  ![image](https://user-images.githubusercontent.com/48917098/204674041-9e096d6a-7714-44e1-a622-d695261f747c.png)
  - input sequence index의 모든 permutation을 고려한 AR 방식을 이용.
    ex) [x1, x2, x3, x4]에 대한 permutation은 4!=24개(Zt=[[1,2,3,4], [1,2,4,3], ... , [4,3,2,1]]).
    이 Zt에 대해 AR LM의 objective를 적용하면 아래와 같음.(mem(m)은 Transformer-XL의 memory state)
    

https://user-images.githubusercontent.com/48917098/204674328-2a990dae-6be5-49de-b4f3-4437ca3401e9.mp4


    - 각 토큰들은 원래 순서에 따라 positional encoding이 부여되기 때문에 토큰들의 상대적 position을 구분할 수 있음.
    - AR 방식이라 양방향 context를 고려하였고, AE 방식의 한계인 pre-training과 fine-tuning의 불일치가 없음(independent assumption이 없고 [mask]도 없기 때문).
    
  ### 2.2 Target-Aware Representation for Transformer
  - Permutation LM(PLM)은 아래와 같은 문제점이 발생할 수 있기 때문에 이 문제를 해결하기 위한 방법을 제시.
  ![image](https://user-images.githubusercontent.com/48917098/204673901-1f48c4a3-e39b-4d98-a44d-0c32c8c3bea0.png)
  - 이 문제점을 해결하기 위해 이전 context 토큰들의 정보(x_(z<t))뿐만 아니라 target index의 position 정보(z_t)도 함께 이용.
  ![image](https://user-images.githubusercontent.com/48917098/204674925-5e759cfc-60d7-4549-86b6-deaf6770efd0.png)
  
  ### 2.3 Two-Stream Self-Attention for Target Aware Representation
  - target position 정보를 추가적으로 이용하기 위한 g의 조건은 다음과 같다.
  <img width="658" alt="image" src="https://user-images.githubusercontent.com/48917098/204675095-4841390b-81d5-470a-934b-803c89bde1d4.png">
  - 위의 두가지 조건을 만족하기 위해 토큰 1개 당 1개의 representation만을 갖는 standard transformer 구조가 아닌, 2개의 hidden representation을 이용하는 변형된 transformer 구조를 제안.
    #### 1. Query Representation
    - 이전 시점 token들의 content와 현재 시점의 위치 정보를 이용하여 계산되는 representation.
    ![image](https://user-images.githubusercontent.com/48917098/204675654-b27dce43-f1d9-4430-a3fb-a855e7e5be4a.png)

    #### 2. Context Representation
    - 현재 시점 및 이전 시점 token들의 content를 이용하여 계산되는 representation.
    - standard transformer의 hidden state와 동일한 역할.
    ![image](https://user-images.githubusercontent.com/48917098/204675837-11502157-26d4-439e-b7ab-abb1708d46c1.png)



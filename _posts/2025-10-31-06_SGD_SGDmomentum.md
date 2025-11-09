---
layout: single
title:  "SGD vs SGD w.모멘텀 비교"
description: "ipynb 파일을 .md 로 변환시켜 올려보는 테스트 글"
date: 2025-10-31
categories: [Deep Learning]
tag: [Optimization, Gradient Descent]
toc: true
author_profile: True
---

# (1) 경사하강법(SGD) 과 (2) 모멘텀을 적용한 SGD 비교



기본 **경사하강법(Stochastic Gradient Descent, SGD)** 과 **모멘텀(Momentum)** 을 적용한 경사하강법의 차이를 PyTorch로 비교

---

## (1) 기본 SGD

- Pytorch 의 'optim' 라이브러리를 이용해서 자동으로 가중치와 편향 계산 및 업데이트

- 텐서 `torch.tensor`와 넘파이 `numpy.array()` 간 변환을 명확하게 구분 할 것



```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
```


```python
# 샘플 데이터 선언
sampleData1 = np.array([
    [166, 58.7],
    [176, 75.7], 
    [171, 62.1],
    [173, 70.4],
    [169, 60.1]
])

# cost function : MSE
def mse(Yp, Y):
    loss = ((Yp - Y)**2).mean()
    return loss

# x, y 만들기
x = sampleData1[:, 0]   # 키
y = sampleData1[:, 1]   # 몸무게

# x, y 값을 평균 기준으로 스케일링 (loss 값 NaN 방지✨)
X = (x - x.mean())
Y = (y - y.mean())

# x,y 변수 -> 텐서로 변환
X = torch.tensor(X)
Y = torch.tensor(Y)

```

### 최적화 함수 사용하기

- 최적화 함수란? `no_grad()` 를 사용한 학습파라미터 (Weight, Bias) 업데이트 과정을 자동으로 처리해줌

> with torch.no_grad():
>    W -= lr* W.grad
>    B -= lr* B.grad

- torch.optim 안에는 GD, SGD, SGD with momentum, Adam, ... 등 다양한 최적화 함수가 존재함! 

```python

# 1. 가중치와 편향 초기값 설정
W = torch.tensor(1., requires_grad = True)
B = torch.tensor(1., requires_grad = True)

# 예측 함수 및 예측값
def pred(X):
    y = W*X +B
    return y

# 2. hyperparmeter 설정 (학습률, 에폭수)
lr = 0.001
num_epochs = 500

optimizer = optim.SGD([W, B], lr= lr)

# 3. 손실 값 저장하는 녀석 만들기
history = np.zeros([0,2])   # row : 0 개, col: 두 개 (epoch, loss(MSE))
```


```python
# 4. 학습 루프
for epoch in range(num_epochs):

    ## 모델의 예측값 계산
    Yp = pred(X)

    ## loss 구하기
    loss = mse(Yp, Y)  # 텐서구조

    ## gradient 계산
    loss.backward()

    ## 계산된 학습 파라미터 자동 업데이트
    optimizer.step()

    ## W, B 초기화
    optimizer.zero_grad()

    # 0, 10, 20, 30,,... 에폭때 저장해라
    if epoch % 10 == 0:
        item = np.array([epoch, loss.item()])  # 텐서에서 넘파이로 접근
        history = np.vstack([history, item])   # 아이템 내용 history 안에 저장
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.3f}")
```

```python
# 5. loss 시각화
plt.plot(history[:, 0], history[:, 1], label= "SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![SGD_Loss](/assets/img/posts/sgd_loss.png)

---

## (2) SGD with Momentum


- 나머지 코드는 동일하고 optimizer 설정만 `momentum = 0.9` 추가하여 다름

- "관성(momentum)"의 개념을 도입해, **경사가 작은 구간에서도 학습이 멈추지 않고 꾸준히 이동할 수 있도록** 도움

- 일반 SGD 와 어떻게 다른지 확인 👇


```python
# 1. Weight, Bias 초기값 설정
W = torch.tensor(1.0, requires_grad = True)
B = torch.tensor(1.0, requires_grad = True)

# 2. 학습 설정 (동일한 학습률과 에폭)
lr = 0.001
num_epochs = 500

#🔸 2.5 momentum 을 추가한 최적화 SGD 함수
optimizer = optim.SGD([W, B], lr= lr, momentum = 0.9)

# 3. 손실 저장용 변수 만들기
# history for SGD w momentum
history2 = np.zeros([0, 2]) # epoch, loss
```


```python
# 4. 학습 루프
for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        item = np.array([epoch, loss.item()])  
        history2 = np.vstack([history2, item])
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
```

```python
# 5. 손실 비교 시각화(SGD 와 SGD w. momentum)
plt.plot(history[:, 0], history[:, 1], label = "SGD")
plt.plot(history2[:, 0], history2[:, 1], label = "SGD w Momentum 0.9")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```
![SGD_Loss_w_Momemtum](/assets/img/posts/sgd_loss_momentum.png)

---

🔹 즉, 결과 요약
| 최적화 함수          | 특징                   | 수렴 속도   |
| ------------------ | -------------------- | ------- |
| **SGD**            | 단순한 기울기 하강 방식        | 느림      |
| **SGD + Momentum** | 이전 업데이트 방향을 고려한 관성 효과 추가 | 빠르고 안정적 |

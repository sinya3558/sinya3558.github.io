---
layout: single
title:  "SGD vs SGD w.모멘텀 비교"
description: "ipynb 파일을 .md 로 변환시켜 올려보는 테스트 글"
date: 2025-10-31
categories: [Deep Learning]
tag: [SGD, Momentum, Optimization, Gradient Descent]
toc: true
author_profile: True
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# (1) 경사하강법(SGD) 과 (2) 모멘텀을 적용한 SGD 비교



기본 경사하강법(Stochastic Gradient Descent, SGD)과 **모멘텀(Momentum)** 을 적용한 경사하강법의 차이를 PyTorch로 비교



## (1) 기본 SGD



- Pytorch 의 'optim' 라이브러리를 이용해서 자동으로 가중치와 편향 계산 및 업데이트

- 텐서'torch.tensor'와 넘파이'numpy.array()' 간 변환을 명확하게 구분 할 것



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
# print(x, y)
# print(X, Y)
```

<pre>
[166. 176. 171. 173. 169.] [58.7 75.7 62.1 70.4 60.1]
tensor([-5.,  5.,  0.,  2., -2.], dtype=torch.float64) tensor([-6.7000, 10.3000, -3.3000,  5.0000, -5.3000], dtype=torch.float64)
</pre>

```python
# 최적화 함수 사용하기
# from torch import optim # GD, SGD, SGD with momentum, Adam, ... 전부 optim 안에 있는 최적화를 위한 함수! 
'''
- 최적화 함수란?
with torch.no_grad():
    W -= lr* W.grad
    B -= lr* B.grad

이걸 자동으로 대신해줌
'''

# 1. 가중치와 편향 초기값 설정
W = torch.tensor(1., requires_grad = True)
B = torch.tensor(1., requires_grad = True)

# 예측 함수 및 예측값
def pred(X):
    y = W*X +B
    return y

# 2. hyperparmeter 설정(lr, # epochs)
lr = 0.001
num_epochs = 500

optimizer = optim.SGD([W, B], lr= lr)

# 3. 값 저장하는 녀석 만들기
history = np.zeros([0,2])   # row : 0 개, col: 두 개 (epoch, loss(MSE))
```


```python
# 4. 학습
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

<pre>
Epoch: 1, Loss: 13.352000000000013
Epoch: 11, Loss: 10.385507595115332
Epoch: 21, Loss: 8.517290788930145
Epoch: 31, Loss: 7.336381106095175
Epoch: 41, Loss: 6.585766150186489
Epoch: 51, Loss: 6.10470543208963
Epoch: 61, Loss: 5.79266371572748
Epoch: 71, Loss: 5.586756762875927
Epoch: 81, Loss: 5.447643447896195
Epoch: 91, Loss: 5.350703629895281
Epoch: 101, Loss: 5.280527209758149
Epoch: 111, Loss: 5.227462141031004
Epoch: 121, Loss: 5.185457619302882
Epoch: 131, Loss: 5.150715754503327
Epoch: 141, Loss: 5.120848272929033
Epoch: 151, Loss: 5.094348526784347
Epoch: 161, Loss: 5.070261659317728
Epoch: 171, Loss: 5.047978542732032
Epoch: 181, Loss: 5.027106757141466
Epoch: 191, Loss: 5.007389971158269
Epoch: 201, Loss: 4.9886571079744835
Epoch: 211, Loss: 4.970790752648647
Epoch: 221, Loss: 4.953707713394186
Epoch: 231, Loss: 4.9373465120754165
Epoch: 241, Loss: 4.9216594573070225
Epoch: 251, Loss: 4.906608406221416
Epoch: 261, Loss: 4.892160531184037
Epoch: 271, Loss: 4.878287585588823
Epoch: 281, Loss: 4.864964077046622
Epoch: 291, Loss: 4.8521665089436805
Epoch: 301, Loss: 4.839873157841796
Epoch: 311, Loss: 4.828063558274893
Epoch: 321, Loss: 4.8167182117834795
Epoch: 331, Loss: 4.805818473117696
Epoch: 341, Loss: 4.795346934846826
Epoch: 351, Loss: 4.785286550614115
Epoch: 361, Loss: 4.775621229881556
Epoch: 371, Loss: 4.766335316285357
Epoch: 381, Loss: 4.757413854810467
Epoch: 391, Loss: 4.7488426372723795
Epoch: 401, Loss: 4.740607840239167
Epoch: 411, Loss: 4.732696256087576
Epoch: 421, Loss: 4.725095186503433
Epoch: 431, Loss: 4.7177925003574845
Epoch: 441, Loss: 4.710776384609207
Epoch: 451, Loss: 4.704035705671187
Epoch: 461, Loss: 4.697559591128709
Epoch: 471, Loss: 4.691337616846559
Epoch: 481, Loss: 4.685359880215156
Epoch: 491, Loss: 4.679616772626916
</pre>

```python
# loss 시각화
plt.plot(history[:, 0], history[:, 1], label= "SGD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

<img src="data:image/png;base64,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GQAAkLUIMgAAIGsRZACMOpZladmyZXaXASAJCDIA0upv/uZvZFnWAY+zzjrL7tIAZCGX3QUAGH3OOussPfDAA4OOeb1em6oBkM3okQGQdl6vVzU1NYMeZWVlkqK3fZYuXar58+crPz9fRx11lJ544olBn1+/fr2+8pWvKD8/XxUVFbrsssvU3d096Jzf/va3mjZtmrxer8aOHaurr756UHtLS4suuOACFRQU6Oijj9by5ctT+6UBpARBBkDGWbx4sS666CK9++67WrBggS655BJt2LBBktTT06MzzzxTZWVleuutt/T444/rhRdeGBRUli5dqquuukqXXXaZ1q9fr+XLl2vy5MmDfsfPfvYzfeMb39B7772nr33ta1qwYIFaW1vT+j0BJMEId/IGgCOycOFC43Q6TWFh4aDHrbfeaowxRpK5/PLLB31m9uzZ5oorrjDGGHPfffeZsrIy093dnWj/05/+ZBwOh2lqajLGGFNbW2uuv/76IWuQZH76058m3nd3dxtJ5umnn07a9wSQHoyRAZB2p59+upYuXTroWHl5eeJ1Q0PDoLaGhgatW7dOkrRhwwbNnDlThYWFifY5c+YoEolo06ZNsixLu3bt0ty5cw9Zw4wZMxKvCwsL5fP51NzcPNyvBMAmBBkAaVdYWHjArZ5kyc/PP6zz3G73oPeWZSkSiaSiJAApxBgZABnnzTffPOD9McccI0k65phj9O6776qnpyfR/vrrr8vhcGjKlCkqLi7WxIkTtWLFirTWDMAe9MgASDu/36+mpqZBx1wulyorKyVJjz/+uE466SSdeuqpevjhh7V69Wr95je/kSQtWLBAN954oxYuXKibbrpJe/fu1fe+9z1961vfUnV1tSTppptu0uWXX66qqirNnz9fXV1dev311/W9730vvV8UQMoRZACk3TPPPKOxY8cOOjZlyhRt3LhRUnRG0aOPPqorr7xSY8eO1SOPPKJjjz1WklRQUKBnn31W11xzjU4++WQVFBTooosu0h133JH4WQsXLlR/f7/+5V/+RT/60Y9UWVmpiy++OH1fEEDaWMYYY3cRABBnWZaefPJJnX/++XaXAiALMEYGAABkLYIMAADIWoyRAZBRuNsN4EjQIwMAALIWQQYAAGQtggwAAMhaBBkAAJC1CDIAACBrEWQAAEDWIsgAAICsRZABAABZ6/8DfB/SMkmGReAAAAAASUVORK5CYII="/>

## (2) SGD with Momentum



- 나머지 코드는 동일하고 optimizer 설정만 `momentum = 0.9` 추가하여 다름

- "가속도(momentum)"의 개념을 도입해, 경사가 작은 구간에서도 꾸준히 이동할 수 있도록 도움

- 일반 SGD 와 어떻게 다른지 확인



```python
# 1. W, B 초기값 설정
W = torch.tensor(1.0, requires_grad = True)
B = torch.tensor(1.0, requires_grad = True)

# 2. 동일한 학습률과 에폭
lr = 0.001
num_epochs = 500

#🔸 2.5 최적화함수(momentum 추가)
optimizer = optim.SGD([W, B], lr= lr, momentum = 0.9)

# 3. 손실 추척하는 녀석 만들기
# history for SGD w momentum
history2 = np.zeros([0, 2]) # epoch, loss
```


```python
# 4. 학습
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

<pre>
Epoch: 1, Loss: 4.5390
Epoch: 11, Loss: 4.5390
Epoch: 21, Loss: 4.5390
Epoch: 31, Loss: 4.5390
Epoch: 41, Loss: 4.5390
Epoch: 51, Loss: 4.5390
Epoch: 61, Loss: 4.5390
Epoch: 71, Loss: 4.5390
Epoch: 81, Loss: 4.5390
Epoch: 91, Loss: 4.5390
Epoch: 101, Loss: 4.5390
Epoch: 111, Loss: 4.5390
Epoch: 121, Loss: 4.5390
Epoch: 131, Loss: 4.5390
Epoch: 141, Loss: 4.5390
Epoch: 151, Loss: 4.5390
Epoch: 161, Loss: 4.5390
Epoch: 171, Loss: 4.5390
Epoch: 181, Loss: 4.5390
Epoch: 191, Loss: 4.5390
Epoch: 201, Loss: 4.5390
Epoch: 211, Loss: 4.5390
Epoch: 221, Loss: 4.5390
Epoch: 231, Loss: 4.5390
Epoch: 241, Loss: 4.5390
Epoch: 251, Loss: 4.5390
Epoch: 261, Loss: 4.5390
Epoch: 271, Loss: 4.5390
Epoch: 281, Loss: 4.5390
Epoch: 291, Loss: 4.5390
Epoch: 301, Loss: 4.5390
Epoch: 311, Loss: 4.5390
Epoch: 321, Loss: 4.5390
Epoch: 331, Loss: 4.5390
Epoch: 341, Loss: 4.5390
Epoch: 351, Loss: 4.5390
Epoch: 361, Loss: 4.5390
Epoch: 371, Loss: 4.5390
Epoch: 381, Loss: 4.5390
Epoch: 391, Loss: 4.5390
Epoch: 401, Loss: 4.5390
Epoch: 411, Loss: 4.5390
Epoch: 421, Loss: 4.5390
Epoch: 431, Loss: 4.5390
Epoch: 441, Loss: 4.5390
Epoch: 451, Loss: 4.5390
Epoch: 461, Loss: 4.5390
Epoch: 471, Loss: 4.5390
Epoch: 481, Loss: 4.5390
Epoch: 491, Loss: 4.5390
</pre>

```python
# 손실 비교 시각화(SGD 와 SGD w. momentum)
plt.plot(history[:, 0], history[:, 1], label = "SGD")
plt.plot(history2[:, 0], history2[:, 1], label = "SGD w Momentum 0.9")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

<img src="data:image/png;base64,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FCNHjhR79+4VK1asEAkJCeKpp54Kxk9q1ZYvXy6+/fZbkZubK3JycsTTTz8t9Hq9OHjwoBCCdR1IO3bsEB07dhS9e/cWs2fP9h5nnSvn+eefFz169BDnz5/3Pi5cuOA9r2ZdM8j4YeDAgWLWrFne906nU6Smpor58+cHsVShr3aQcblcIjk5Wbz22mveY8XFxcJoNIrPP/9cCCHEoUOHBACxc+dO7zUrV64UkiSJs2fPqlb2UFRQUCAAiI0bNwoh5LrV6/Vi6dKl3muys7MFALF161YhhBw8NRqNyMvL816zcOFCERUVJaxWq7o/IATFxsaKDz74gHUdQGVlZaJz585izZo1YtiwYd4gwzpX1vPPPy/69OlT7zm165pdS81ks9mwe/dujBw50ntMo9Fg5MiR2Lp1axBL1vYcP34ceXl5PnUdHR2NQYMGeet669atiImJQf/+/b3XjBw5EhqNBtu3b1e9zKGkpKQEABAXFwcA2L17N+x2u099d+vWDenp6T713atXLyQlJXmvGTNmDEpLS/HTTz+pWPrQ4nQ6sWTJElRUVGDIkCGs6wCaNWsWJkyY4FO3AP/5DoTDhw8jNTUVV111FaZOnYpTp04BUL+u2/ymkUq7ePEinE6nT+UDQFJSEn7++ecglaptysvLA4B669pzLi8vD4mJiT7ndTod4uLivNdQXS6XC4899hiGDh2Knj17ApDr0mAwICYmxufa2vVd35+H5xz5OnDgAIYMGQKLxYKIiAh89dVX6N69O7KysljXAbBkyRLs2bMHO3furHOO/3wra9CgQVi0aBG6du2K8+fP44UXXsANN9yAgwcPql7XDDJEV6BZs2bh4MGD+PHHH4NdlData9euyMrKQklJCb744gtMnz4dGzduDHax2qTTp09j9uzZWLNmDcLCwoJdnDZv3Lhx3te9e/fGoEGDkJGRgX//+98wmUyqloVdS82UkJAArVZbZ/R1fn4+kpOTg1SqtslTn43VdXJyMgoKCnzOOxwOFBUV8c+jAY8++ij+85//YP369ejQoYP3eHJyMmw2G4qLi32ur13f9f15eM6RL4PBgE6dOqFfv36YP38++vTpg7///e+s6wDYvXs3CgoKcO2110Kn00Gn02Hjxo146623oNPpkJSUxDoPoJiYGHTp0gVHjhxR/Z9vBplmMhgM6NevH9atW+c95nK5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"/>

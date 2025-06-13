# vibEGG

## 데이터 전처리 과정
1. 폴더명을 순서에 따라 정렬하고 레이블 딕셔너리를 생성
2. 각 데이터 내에서 전체 채널의 최대 최소값으로 정규화
3. (3) 채널 시계열 데이터 (4) 개를 한 배치로 묶음
4. 한 배치 내에서 가장 긴 시퀀스 길이로 패딩함

## 모델 구조
기존 vibMilk 논문을 참고하여 작성



### `Inception Module`

kerner_size = k, stride = s, padding = p 인 Conv1d는 `Conv1d(k,s,p)` 로 표기함

1. BottleNeck

입력의 $ \frac{1}{4} $ 의 채널을 갖도록 `Conv1d(1,1,0)`로 구성된 BottleNeck 레이어를 통과하여 BT에 저장한다.   
이때 입력의 1/4이 0일 경우 기존 입력 차원과 동일하게 유지하고 BottleNeck는 nn.Identitiy로 대체한다.


2. Conv1d

`Conv1d(5,1,2), Conv1d(9,1,4), Conv1d(17,1,8)` 에 BT를 통과시켜 x1, x2, x3에 저장한다.   
`Maxpool(3,1,1)` 과 `Conv1d (1,1,0)` 에 차례로 BT를 통과시켜 x4에 저장한다.


3. Concat

각각 출력 차원 (출력 채널 수) 의 $ \frac{1}{4} $ 채널 수를 갖는 x1, x2, x3, x4를 채널 차원으로 `Concat`연산 취하여 out에 저장한다.


4. 출력 

동일 차원으로 배치 정규화 후 ReLU 층 통과시켜 출력한다.



### `Residual Block`

1. Main Flow

3개의 `Inception Module`을 연달아 통과시킨다.


2. Short Cut

처음 입력된 텐서에 대해 Residual Block의 출력 형상과 맞추는 short cut 연산을 시행한다.


3. Residual Skip

Main Flow 후 텐서와 Short cut 텐서를 원소별로 덧셈하여 출력한다.



### `Entire Model Network`

위에서 설명한 Residual Block 두개와 3층의 FC layer를 통과한다.

1. Residual Block

입력 채널 (3)에서 출력 채널 64를 가지는 첫 번째 Residual Block을 거친다.   
입력 채널 64에서 출력 채널 128을 가지는 두 번째 Residual Block을 거친다.

2. Pooling & Flatten

(sequence_len, channels)의 각 데이터에 대해, 각 채널에 해당하는 시계열 값들의 평균 값인 하나의 값으로 풀링한다.   
풀링한 데이터를 128 길이의 FC레이어 입력으로 Flatten한다.

3. FC layer

입력 차원 128에서 출력 차원 64를 가지는 첫 번째 FC layer를 거친다.   
ReLU를 통과한다.   
입력 차원 64에서 출력 차원 32를 가지는 두 번째 FC layer를 거친다.   
ReLU를 통과한다.   
입력 차원 32에서 출력 차원 `num_classes`를 가지는 세 번째 FC layer를 거친다.   
softmax를 거쳐 최종적으로 각 레이블에 해당할 확률을 출력한다.





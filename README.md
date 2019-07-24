# brunch_nafma

브런치 사용자를 위한 글 추천 대회 https://arena.kakao.com/c/2 
에 참가한 NAFMA 팀의 소스코드를 담은 리파지토리입니다.  


## 1. 모델 입력 데이터 생성
독자가 읽은 글과 작가가 생성한 글의 키워드를 이용하여 문장을 생성합니다.

```bash
$> python prepare_d2v.py
``` 

## 2. 모델 학습 (Doc2Vec)
tensorflow 를 이용하여 Doc2Vec을 구현하였습니다. 

```bash
$> python train.py
``` 

모델의 크기는 336MB 이고 분할 압축되어 저장되어 있습니다. 
아래의 명령어로 압축해제 할 수 있습니다. 

```bash
$> cd model
$> cat model.tar.gza* | tar xvfz -
``` 


## 3. 평가 데이터 생성 

```bash
$> python inference.py
``` 

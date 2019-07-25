# brunch_nafma

브런치 사용자를 위한 글 추천 대회 https://arena.kakao.com/c/2 
에 참가한 NAFMA 팀의 소스코드를 담은 리파지토리입니다.  



## 0. 데이터 다운로드
https://arena.kakao.com/c/2/data 에 제공된 파일을 
res 디렉토리에 다운로드 받습니다. 

```bash
├── res
   ├── /contents
   ├── /predict
   ├── /read
   ├── magazine.json
   ├── metadata.json
   └── users.json
``` 

## 1. 모델 입력 데이터 생성
독자가 읽은 글들과 작가가 생성한 글의 키워드를 이용하여 문장을 생성합니다.
키워드로만 모델을 만든 이유는 tokenizing, stopword removal 등 전처리 과정이 단순해지기 때문입니다.
(전처리에 필요한 konlpy를 쓰는 것이 불이익이 생길 수 있다니 ...)

```bash
$> python prepare_d2v.py
``` 

## 2. 모델 학습 (Doc2Vec)
tensorflow 를 이용하여 Doc2Vec을 구현하였습니다.
읽은 글들로 추정되는 독자와 작성한 글들로 추정되는 작가간의 유사도를 추천에 반영하기 위한 모델입니다.
개별 단어의 embedding에는 word2vec이 쓰이고 문장, paragraph, 문서 등의 보다 큰 단위의 텍스트를 embedding하는 데에는 doc2vec 기술이 쓰여집니다.
일반적으로 gensim을 사용하는데 라이센스 문제로 쓸 수 없는 상황이라 tensorflow를 사용하였습니다.

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

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
생성 결과 파일은 ./res/writer_user_sentences_keyword.txt 입니다.

## 2. 모델 학습 (Doc2Vec)
Doc2Vec은 읽은 글들로 추정되는 독자와 작성한 글들로 추정되는 작가간의 유사도를 추천에 반영하기 위해 필요한 모델입니다.
단어의 embedding에 쓰이는 모델이 word2vec이고 문단, 문서 등의 보다 큰 단위의 텍스트를 embedding하는 데 쓰이는 모델이 doc2vec입니다.  

독자와 작가를 embedding하는 방법에는 다음과 같은 두가지가 있을 수 있는데 전자는 3월 이후의 글에 적용할 수 없어서 후자를 사용하였습니다.  
1. 각 글을 단어로 보고 독자가 읽은 글의 sequence와 작가가 작성한 글의 sequence를 문장으로 보아서 embedding
2. 자연어 단어로 각 글의 대표 sentence를 준비하고 (독자id + 읽은글 sentence), (작가id + 작성한글 sentence) 로 embedding

일반적으로 gensim doc2vec이 사용되는데 라이센스 문제로 쓸 수 없는 상황이라 tensorflow를 사용하여 구현하였습니다.

```bash
$> python train.py
``` 

train.py의 최종 결과물은 tensorflow model 파일과 embedding vector 파일입니다.

tensorflow model 은 model 디렉토리에 저장됩니다. 
모델의 크기는 336MB 이고 분할 압축되어 저장되어 있습니다.  (github 용량 제한 때문에..)
아래의 명령어로 압축해제 할 수 있습니다. 

```bash
$> cd model
$> cat model.tar.gza* | tar xvfz -
``` 

embedding vector 파일은 ./doc_embeddings_keyword.npy 입니다. 
inference.py 에서 해당 vector를 이용하여 독자와 작가간의 유사도를 계산합니다. 


## 3. 평가 데이터 생성 
inference.py을 실행하면 ./recommend.txt 결과 파일이 생성됩니다.

```bash
$> python inference.py
``` 

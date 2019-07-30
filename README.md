# brunch_nafma

브런치 사용자를 위한 글 추천 대회 https://arena.kakao.com/c/2 
에 참가한 NAFMA 팀의 소스코드를 담은 리파지토리입니다.  


## 1. 실행
### 1.0. 데이터 다운로드
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

### 1.1. 학습 데이터 생성 (Doc2Vec)
주어진 글들의 메타데이타인 metadata.json과 ./res/read에 들어있는 조회데이터들을 이용하여 학습데이터를 준비합니다. 
Doc2Vec 모델을 학습시키기 위하여 독자가 읽은 글의 키워드, 작가가 작성한 글의 키워드로 학습데이터를 만들어냅니다. 자세한 설명은 2장에 있습니다.

```bash
$> python prepare_d2v.py
``` 
생성 결과 파일은 ./res/writer_user_sentences_keyword.txt 입니다.  
중복 조회를 제거하는 처리(리스트를 세트로 변환 L100, L102, L119, L121)에서 랜덤성이 있어서 실행시마다 독자가 읽은 글의 키워드의 순서가 바뀌어 나올 수 있습니다. 키워드의 내용은 같고 순서만 달라지므로 성능에 별로 영향이 없습니다..

### 1.2. 모델 학습 (Doc2Vec)

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


### 1.3. 평가 데이터 생성 
inference.py을 실행하면 ./recommend.txt 결과 파일이 생성됩니다.

```bash
$> python inference.py
``` 

## 2. 알고리즘
좋은 알고리즘을 만들려면 서비스와 데이터를 잘 이해해야 합니다 ^^  
브런치를 많이 사용해보아야 하고 브런치에서 제공한 글들이 서비스와 데이터를 이해하는데에 큰 도움이 됩니다.
* https://brunch.co.kr/@kakao-it/332 브런치 데이터
* https://brunch.co.kr/@kakao-it/333 브런치의 추천 소개
* https://brunch.co.kr/@kakao-it/72 브런치의 추천 기술

구현한 알고리즘을 간단하게 설명하자면 아래와 같습니다.  
1. 독자의 작가에 대한 선호도, 독자 성향과 작가 성향간의 유사도를 계산하여 테스트 독자가 소비할 글을 예측한다.
2. 대다수 독자의 글 소비 패턴을 활용하여 특정한 테스트 독자가 소비할 글을 예측한다.

전자는 content-based recommendation에 해당하고 후자는 collaborative recommendation에 해당합니다.
각각에 대해서 좀더 상세하게 설명하겠습니다.

### 2.1. Content-based (preference-based) recommendation
Content-based recommendation이란 컨텐츠가 가지고 있는 속성을 매개로 하여 추천하는 것으로 간단한 예를 들자면 여행을 좋아하는 독자에게 여행에 관한 글을 추천하는 것입니다. 독자가 여행을 좋아하는지를 판별하기 위해서는 독자가 읽은 글들의 내용을 보면 알 수 있는데 다행하게 글의 내용을 요약한 키워드가 메타데이터로 존재하므로 키워드에 여행이라는 단어의 존재 여부를 따지면 됩니다. 여행에 관한 글 또는 작가를 판별하는 것도 마찬가지로 작성된 글의 키워드를 이용하면 됩니다. 정리하면, 글의 키워드는 컨텐츠가 가지고 있는 속성에 해당하고 그것을 매개로 독자에게 글 또는 작가를 추천할 수 있습니다.  
글의 소비가 작가 단위로 이루어지는 경우가 많기 때문에 여기서는 작가 단위로 판별하는 방법을 사용하였으며 구체적으로는 아래에 설명하는 Doc2Vec 모델을 사용하였습다.  
이외로 content-based의 개념을 확장하여 독자의 작가에 대한 preference를 최대한 활용하여 추천을 하였습니다. 구독 여부와 조회 여부가 독자의 작가에 대한 preference를 보여주는 데이터입니다.

### 2.1.1 Doc2Vec 모델
숫자가 아닌 데이터를 숫자로 바꾸어서 수식에 사용하는 기법을 머신 러닝에서 vector embedding이라고 합니다.
텍스트 데이터의 경우 단어의 embedding에 쓰이는 모델이 word2vec이고 문단, 문서 등의 보다 큰 단위의 텍스트를 embedding하는 데 쓰이는 모델이 doc2vec입니다.  
독자와 작가의 성향의 유사도를 계산하기 위해서는 독자와 작가를 vector embedding 해야 합니다.
독자는 읽은 글들로 특징화 할 수 있고 작가는 작성한 글들로 특징화 할 수 있으므로 위의 문단, 문서 데이터를 embedding하는 doc2vec 모델을 사용할 수 있습니다. 
다시 말하자면 한 명의 독자는 하나의 문서가 되는 것이고 각 문서는 각 독자가 읽은 글들을 텍스트로 구성합니다. 마찬가지로 한 명의 작가는 하나의 문서가 되는 것이고 각 문서는 각 작가가 작성한 글들을 텍스트로 구성합니다. 

독자가 읽은 글들과 작가가 작성한 글의 키워드를 이용하여 문서에 넣을 sentence를 준비하였는데 키워드로만 모델을 만든 이유는 tokenizing, stopword removal 등 전처리 과정이 단순해지기 때문입니다. 그래도 중복 제거 등의 일부 전처리는 여전히 필요합니다.

처음에는 일반적으로 많이 쓰이는 gensim doc2vec을 사용했으나 라이센스 문제로 쓸 수 없는 상황임을 인지하여 tensorflow를 사용하여 변경 구현하였습니다.
https://github.com/sachinruk/doc2vec_tf  (Apache 2.0 라이센스) 코드를 활용하였습니다. 

### 2.2. Collaborative recommendation
Collaborative recommendation이란 다른 사용자들의 조회 패턴을 이용하여 추천하는 것으로 아마존 또는 넷플릭스 추천의 기본 알고리즘이고 구체적으로는 collaborative filtering 또는 CF라고 불립니다. 본 문제에 일반적인 CF를 사용하면 낮은 성능을 보일 것으로 예상하였는데 그 이유는 사용자의 조회 패턴이 시간의 흐름에 의존적이기 때문입니다. 즉, sequence-aware recommendation 또는 session-based recommendation을 해야 하는 문제라고 판단하였습니다. Session-based recommendation을 하기 위하여 아래의 두가지 방법을 모두 구현하고 둘 중에서 성능이 높은 statistics-based recommendation을 채택하였습니다.
* 개별 글의 id를 단어로 보고 session sentence를 준비하고 word2vec 모델을 만들어서 co-occerence 기반으로 closest article을 추천
* 연이어 조회되는 글들에 대한 statistics를 준비하고 most frequent consecutive article을 추천




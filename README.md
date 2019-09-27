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
Doc2Vec 모델을 학습시키기 위하여 독자가 읽은 글의 키워드, 작가가 작성한 글의 키워드로 학습데이터를 만들어냅니다. 자세한 설명은 2장에 작성하였습니다.

```bash
$> python prepare_d2v.py
``` 
생성 결과 파일은 ./res/writer_user_sentences_keyword.txt 입니다.  
중복 조회를 제거하는 처리(리스트를 세트로 변환 L100, L102, L119, L121)에서 랜덤성이 있어서 (실수죠!) 실행시마다 독자가 읽은 글의 키워드의 순서가 바뀌어 나올 수 있습니다. 하지만 키워드의 내용은 같고 순서만 달라지므로 성능에 별로 영향이 없습니다.

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
문제를 해결할 좋은 알고리즘을 만들려면 서비스와 데이터를 잘 이해해야 합니다 ^^  
브런치를 많이 사용해보아야 하고 카카오에서 제공한 브런치 글들이 서비스와 데이터를 이해하고 문제를 해결하는데에 큰 도움이 됩니다.
* https://brunch.co.kr/@kakao-it/332 브런치 데이터
* https://brunch.co.kr/@kakao-it/333 브런치의 추천 소개
* https://brunch.co.kr/@kakao-it/72 브런치의 추천 기술

본 대회에서 주어진 조회 기록은 2018/10/01 ~ 2019/2/28 까지이고 예측해야 하는 조회의 기간은 2019/2/22 ~ 2019/3/14 입니다.
중요하게 생각할 점은 2/28 까지 작성된 글은 조회 기록을 기반으로 예측할 수 있지만 3/1 이후에 작성된 글은 조회 기록을 알 수 없으므로 일반적인 조회 패턴과 독자, 작가, 글 등의 메타 데이터를 이용하여 예측해야 한다는 점입니다.

이렇게 문제를 이해하고 구현한 알고리즘을 간단하게 설명하자면 아래와 같습니다.  
1. 3/1 이후 작성된 글에 대해서는 독자의 작가에 대한 선호도, 독자 성향과 작가 성향간의 유사도를 계산하여 테스트 독자가 소비할 글을 예측한다.
2. 2/28 까지 작성된 글에 대해서는 대다수 독자의 글 소비 패턴을 활용하여 특정한 테스트 독자가 소비할 글을 예측한다.

전자는 content-based recommendation에 해당하고 후자는 collaborative recommendation에 해당합니다.
각각에 대해서 좀더 상세하게 설명하겠습니다.

### 2.1. Content-based (preference-based) recommendation
Content-based recommendation이란 컨텐츠가 가지고 있는 속성을 매개로 하여 추천하는 것으로 간단한 예를 들자면 여행을 좋아하는 독자에게 여행에 관한 글을 추천하는 것입니다. 독자가 여행을 좋아하는지를 판별하기 위해서는 독자가 읽은 글들의 내용을 보면 알 수 있는데 다행하게 글의 내용을 요약한 키워드가 메타데이터로 존재하므로 키워드에 여행이라는 단어의 존재 여부를 따지면 됩니다. 여행에 관한 글 또는 작가를 판별하는 것도 마찬가지로 작성된 글의 키워드를 이용하면 됩니다. 정리하면, 글의 키워드는 컨텐츠가 가지고 있는 속성에 해당하고 그것을 매개로 독자에게 글 또는 작가를 추천할 수 있습니다.  
글의 소비가 작가 단위로 이루어지는 경우가 많기 때문에 본 문제에 대해 작가 단위로 판별하는 방법을 사용하였으며 구체적으로는 아래에 설명하는 Doc2Vec 모델을 사용하였습다.  
읽거나 작성한 글의 내용으로 추출할 수 있는 속성 외에 글의 신규성과 독자의 작가에 대한 preference가 활용해야 하는 속성입니다. 구독 여부와 조회 여부가 독자의 작가에 대한 preference를 나타내는 데이터입니다. 앞에 언급한 카카오의 브런치 추천 글을 읽어보면 신규성과 구독 여부가 읽을 글을 선택하는 데에 매우 큰 영향을 준다는 것을 알 수 있습니다.

### 2.1.1 Doc2Vec 모델
숫자가 아닌 데이터를 숫자(벡터)로 바꾸어서 수식 모델에 사용하는 기법을 머신 러닝에서 vector embedding이라고 합니다. 데이터를 벡터로 바꾸고 나면 내적 또는 cosine 값을 이용하여 유사도를 계산할 수 있습니다.
텍스트 데이터의 경우 단어의 co-occurence를 이용하여 embedding하는 모델이 word2vec이고 같은 개념을 문장, 문단, 문서 등의 보다 큰 단위의 텍스트에 적용하여 embedding하는 데 쓰이는 모델이 doc2vec입니다.

독자와 작가의 성향의 유사도를 계산하기 위해서는 독자와 작가를 vector embedding 해야 합니다.
독자는 읽은 글의 집합으로 특징화 할 수 있고 작가는 작성한 글의 집합으로 특징화 할 수 있으므로 doc2vec 모델을 사용할 수 있습니다.
다시 말하자면 한 명의 독자는 하나의 문서가 되는 것이고 각 독자 문서는 해당 독자가 읽은 글들의 내용 또는 제목 또는 키워드 텍스트로 구성됩니다. 마찬가지로 한 명의 작가는 하나의 문서가 되는 것이고 각 작가 문서는 해당 작가가 작성한 글들의 내용 또는 제목 또는 키워드 텍스트로 구성됩니다.

저희 팀은 독자가 읽은 글들과 작가가 작성한 글의 키워드만을 이용하여 문서에 넣을 sentence를 준비하였는데 키워드로만 모델을 만든 이유는 tokenizing, stopword removal 등 전처리 과정이 단순해지기 때문입니다. 그래도 중복 제거 등의 일부 전처리는 여전히 필요합니다.
시간의 제약으로 doc2vec만을 이용하였습니다만 doc2vec외에 word2vec의 weighted average를 사용하고 비교해보는 것이 바람직합니다. doc2vec의 경우에는 학습 데이터의 종류와 분량에 따라 성능이 나쁠 수도 있기 때문입니다. 

처음에는 일반적으로 많이 쓰이는 gensim doc2vec을 사용했으나 라이센스 문제로 쓸 수 없는 상황임을 인지하여 tensorflow를 사용하여 변경 구현하였습니다.
https://github.com/sachinruk/doc2vec_tf  (Apache 2.0 라이센스) 코드를 활용하였습니다.

Doc2Vec 모델은 사용자가 읽을 것으로 예상되는 글 후보를 결정하는 데에도 쓰일 수 있지만 다른 방식으로 글 후보를 뽑은 다음 글 후보의 순서를 결정하는 데에도 쓰일 수 있습니다.

### 2.2. Collaborative recommendation
Collaborative recommendation이란 다른 사용자들의 조회 패턴을 이용하여 추천하는 것으로 아마존 또는 넷플릭스 추천의 기본 알고리즘이고 구체적으로는 collaborative filtering 또는 CF라고 불립니다. 본 문제에 일반적인 CF 방식인 KNN 또는 MF 모델을 사용하면 낮은 성능을 보일 것으로 판단하였는데 그 이유는 조회 데이터로 구성한 user-item matrix가 너무 sparse하고 또한 사용자의 조회 패턴이 시간의 흐름에 의존적이기 때문입니다. 따라서, sequence-aware recommendation 또는 session-based recommendation을 해야 하는 문제라고 판단하였습니다. Session-based recommendation을 하기 위하여 아래의 두가지 방법을 모두 구현해보고 둘 중에서 성능이 높은 statistics-based recommendation을 채택하였습니다.
* 개별 글의 id를 단어로 보고 session sentence를 준비하고 word2vec 모델을 만들어서 co-occerence 기반 similarity를 계산하여 closest article을 추천
* 연이어 조회되는 글들에 대한 statistics를 준비하고 most frequent consecutive article을 추천




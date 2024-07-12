## 여러가지 버전의 chatbot을 만들어 봅시다.

### 1. chatbot1 : 코사인유사도를 이용한 챗봇
* 사전학습된 한국어 SentenceBERT 모델 생성
  : model = SentenceTransformer('ddobokki/klue-roberta-base-nli-sts') 

* 사용자한 질문 문장 --> 문장 임베딩 벡터 구하기
  : embedding = model.encode(질문)

* 코사인유사도 계산 --> 코사인유사도 값을 컬럼(score)으로 저장
  : df_chat['score'] = df_chat.loc[:,'embeddings'].apply(lambda x: cos_sim(x,embedding)).values
    
* 코사인유사도 값이 가장 큰 질문 샘플을 찾아서 해당질문과 짝이되는 답변 추출


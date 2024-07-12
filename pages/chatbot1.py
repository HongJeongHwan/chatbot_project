# 필요한 라이브러리 임포트
import numpy as np
from numpy import dot # 벡터의 내적을 구하기
from numpy.linalg import norm
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from time import sleep

# 사용자정의함수
def cos_sim(A,B):
    return dot(A,B) / (norm(A)*norm(B))

# 사용자정의함수 생성
def chatbot(question):

    # 사용자 질문 문장 --> 문장 임베딩 벡터 구하기
    embedding = model.encode(question)

    # 코사인유사도 계산 --> 코사인유사도 값을 컬럼(score)으로 저장
    df_chat['score'] = df_chat.loc[:,'embeddings'].apply(lambda x: cos_sim(x,embedding)).values
    
    # 코사인유사도 값이 가장 큰 질문 샘플을 찾아서 해당질문과 짝이되는 답변 추출
    # score = df_chat['score'].sort_values(ascending=False).iloc[0]
    # cond = df_chat.loc[:,'score'].values == score
    # result = df_chat.loc[cond,'A'].values[0]
    result = df_chat.loc[df_chat['score'].idxmax(),'A']

    return result

st.image('./data/chatbot1/smile.webp')

# 로딩바 구현하기
with st.spinner(text="페이지 로딩중입니다. 잠시만 기다려 주세요..."):
    sleep(10)


# # 문장 임베딩 생성용 데이터 불러오기
file_path = './data/chatbot1/ChatBotData.csv'
df_chat = pd.read_csv(file_path)

# 사전학습된 한국어 SentenceBERT 모델 생성
model = SentenceTransformer('ddobokki/klue-roberta-base-nli-sts')

# # 테스트 문장 생성
# sentence = "본 발명은 바위수염 추출물을 포함하는 항비만용 조성물 및 그 제조방법에 관한 것이다"
# embedding = model.encode(sentence)
# # st.write(embedding)

# # 임베딩 벡터 생성
# embeddings = df_chat.loc[:,'Q'].apply(lambda x: model.encode(x)).values
# df['embeddings'] = embeddings

# 미리 생성해둔 임베딩 벡터 불러오기
file_path = './data/chatbot1/chatbot_embeddings.npy'
embeddings = np.load(file_path, allow_pickle=True)
df_chat['embeddings'] = embeddings

# 사용자 입력
# text = input()
text = st.text_input('저는 말하고 싶은 챗봇입니다. 저에게 말을 하고싶은 말을 해주세요..', '')

# 챗봇의 답변
if text !='':
    st.write(chatbot(text))

# ans = chatbot(text)
# st.write(ans)
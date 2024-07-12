import streamlit as st

st.header('여러종류의 chatbot test 하기')
st.subheader('1. chatbot1 : 코사인유사도를 이용한 챗봇')
st.write('* 사전학습된 한국어 SentenceBERT 모델 생성')
st.write('  : model = SentenceTransformer("ddobokki/klue-roberta-base-nli-sts")')
st.write('* 사용자한 질문 문장 --> 문장 임베딩 벡터 구하기')
st.write('  : embedding = model.encode(질문)')
st.write('* 코사인유사도 계산 --> 코사인유사도 값을 컬럼(score)으로 저장')
st.write('  : df_chat["score"] = df_chat.loc[:,"embeddings"].apply(lambda x: cos_sim(x,embedding)).values')
st.write('* 코사인유사도 값이 가장 큰 질문 샘플을 찾아서 해당질문과 짝이되는 답변 추출')




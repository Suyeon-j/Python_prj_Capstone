# streamlit run langst.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
GOOGLE_API_KEY=' '
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

@st.cache_resource
def load_model():
    loader = PyPDFLoader("./Lang/mbti.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ['GOOGLE_API_KEY'])
    vector_store = Chroma.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_template = """
    Use the following pieces of context from a document to predict the user's MBTI and explain its features.
    Based on the given summaries and user's input, determine the most likely MBTI type and provide a detailed explanation of its characteristics, strengths, weaknesses, and suitable careers. 
    If the input does not provide enough information to make a prediction, just say that "I don't know", don't try to make up an answer.
    ----------------
    {summaries}

    You MUST answer in Korean and in Markdown format:
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}

    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return chain

def chatbot():
    chain = load_model()
    if st.sidebar.button("챗봇 초기화"):
        st.session_state["messages"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("당신의 MBTI를 예측해드립니다. 취미를 입력해보세요! ex) 나는 영화랑 달리기가 좋아")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("MBTI 예측 중..."):
            response = chain.invoke({"question": prompt})
            answer = response['answer']

        st.session_state["messages"].append({"role": "ai", "content": answer})

        with st.chat_message("ai"):
            st.markdown(answer)
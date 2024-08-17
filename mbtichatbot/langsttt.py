# streamlit run laststtt.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import speech_recognition as sr
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
    Based on the given summaries, chat history, and user's input, predict the user's mbti and explain the characteristics of the mbti. 
    If there is not enough information in the input, ask additional questions for prediction to receive additional inputs at least 3 times and make predictions.
    If you cannot predict even if you include additional inputs, don't make an answer and answer that "I don't know".
    ----------------
    {summaries}
    {chat_history}

    You MUST answer in Korean and in Markdown format:
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    memory = ChatMessageHistory() 
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return chain, memory

def msg(prompt):
    if "chain" not in st.session_state:
        st.session_state["chain"], st.session_state["memory"] = load_model()
        
    chain = st.session_state["chain"]
    memory = st.session_state["memory"]
    
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    memory.add_message(HumanMessage(content=prompt))

    with st.spinner("MBTI 예측 중..."):
        response = chain.invoke({"question": prompt, "chat_history": memory.messages})
        answer = response['answer']

    st.session_state["messages"].append({"role": "ai", "content": answer})
    memory.add_message(AIMessage(content=answer))
    with st.chat_message("ai"):
        st.markdown(answer)
    
def chatbot():
    if st.sidebar.button("챗봇 초기화"):
        st.session_state["messages"] = []
        if "chain" in st.session_state:
            del st.session_state["chain"]
            del st.session_state["memory"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    print(st.session_state["messages"])
    
    button = st.sidebar.button("음성인식")
    prompt = st.chat_input("당신의 MBTI를 예측해드립니다. 취미를 입력해보세요! ex) 나는 영화랑 달리기가 좋아")
    if button:
        Recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            Recognizer.adjust_for_ambient_noise(source)
            st.write("📢 말씀해주세요.")
            audio = Recognizer.listen(source)
        try:
            data = Recognizer.recognize_google(audio, language="ko")
        except:
            st.write("다시 시도해보세요.")
            data = None
            
        prompt = data
        if prompt:
            msg(prompt)
        
    else:
        if prompt:
            msg(prompt)
            
if __name__ == "__main__":
    chatbot()

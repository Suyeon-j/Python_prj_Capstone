# streamlit run langstt.py
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
import speech_recognition as sr

GOOGLE_API_KEY = ' '
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def load_pdf(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

@st.cache_resource
def load_model():
    pdf_paths = ["./Lang/mbti.pdf", "./Lang/MBTI_성격_테스트.pdf"]
    documents = load_pdf(pdf_paths)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ['GOOGLE_API_KEY'])
    vector_store = Chroma.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_template = """
    아래의 문서 내용과 사용자의 질문을 바탕으로, 가장 적합한 MBTI 유형을 예측하고 해당 유형의 특징, 장점, 단점, 그리고 적합한 직업군에 대해 상세히 설명해 주세요.
    사용자의 이전 대화 내용과 현재 질문을 고려하여, MBTI를 정확하게 예측하고 답변해 주시기 바랍니다.
    
    왜 이런 답변이 나왔는지 이유도  답변해주시기 바랍니다.
    문서의 요약 및 사용자의 질문을 고려하여 답변해 주시기 바랍니다. 부족한 정보가 있을 경우, '잘 모르겠습니다'라고 답해 주세요.

    답변은 존댓말로 작성하시고, Markdown 형식을 사용해 주세요.
    ----------------
    {summaries}

    사용자의 질문: {question}

    답변:
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{context}\n\n{question}")
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
        st.session_state["context"] = ""

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["context"] = ""

    if st.sidebar.button("음성 인식"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            st.write("말씀해 주세요.")
            audio = recognizer.listen(source)
        try:
            data = recognizer.recognize_google(audio, language="ko")
            st.session_state["prompt"] = data
        except sr.UnknownValueError:
            st.write("음성을 인식하지 못했습니다. 다시 시도해 주세요.")
        except sr.RequestError:
            st.write(" 실패. 다시 시도해 주세요.")

    if st.session_state["messages"]:
        with st.expander("이전 대화 내용 보기"):
            for msg in st.session_state["messages"]:
                role = "사용자" if msg["role"] == "user" else "챗봇"
                st.markdown(f"**{role}:** {msg['content']}")
        st.markdown("---")

    prompt = st.chat_input("당신의 MBTI를 예측해드립니다. 취미를 입력해보세요! ex) 저는 영화랑 달리기를 좋아합니다.")
    if "prompt" in st.session_state and st.session_state["prompt"]:
        prompt = st.session_state["prompt"]
        del st.session_state["prompt"]

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["context"] = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("MBTI 예측 중..."):
            response = chain.invoke({"context": st.session_state["context"], "question": prompt})
            answer = response['answer']

        st.session_state["messages"].append({"role": "ai", "content": answer})

        with st.chat_message("ai"):
            st.markdown(answer)

if __name__ == "__main__":
    chatbot()
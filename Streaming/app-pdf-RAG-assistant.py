import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# app config
st.set_page_config(page_title="Chat with docs", page_icon="🤖")
st.title("Chat with docs!")


def setup_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    loader = PyPDFLoader("spice.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    docs = loader.load_and_split(
        text_splitter=text_splitter
    )
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="db"
    )
    return vectorstore


def create_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = ChatOllama(model="mistral", straming=True)
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    retriever = db.as_retriever()
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the above conversation, generate a search query to look up in order to get information relevant to "
         "the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    return retrieval_chain


def process_chat(chat_chain, question, chat_history):
    rag_chain = chat_chain.pick("answer")
    result = rag_chain.stream({
        "chat_history": chat_history,
        "input": question,
    })
    return result


chain = create_chain()
chat_history = [
    AIMessage(content="I am here to assist you, ask me anything from the document!"),
]

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = chat_history

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_input = st.chat_input("Ask me anything...")

if user_input:
    chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        response = st.write_stream(process_chat(chain, user_input, chat_history))
    chat_history.append(AIMessage(content=response))
    st.session_state.chat_history = chat_history

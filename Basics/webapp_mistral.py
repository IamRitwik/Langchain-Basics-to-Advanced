from langchain_community.chat_models import ChatOllama
import streamlit as st

llm = ChatOllama(model="mistral")

st.title("Ask me anything!")

question = st.text_input("Enter the question: ")

if question:
    response = llm.invoke(question)
    st.write(response.content)

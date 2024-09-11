from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="mistral")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

firstChain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])
secondChain = speech_prompt | llm
finalChain = firstChain | secondChain

st.title("Speech Generator!")

topic = st.text_input("Enter the topic: ")

if topic:
    response = finalChain.invoke({"topic": topic})
    st.write(response.content)

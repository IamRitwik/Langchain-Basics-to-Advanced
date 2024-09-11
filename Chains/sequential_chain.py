from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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
    input_variables=["title", "emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words
    for the following title: {title}
    Format the output with 2 keys as json format: 'title', 'speech' and fill them 
    with respective values
    """
)

firstChain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])
secondChain = speech_prompt | llm | JsonOutputParser()
finalChain = firstChain | (lambda title: {"title": title, "emotion": emotion}) | secondChain

st.title("Speech Generator!")

topic = st.text_input("Enter the topic: ")
emotion = st.text_input("Enter the emotion: ")

if topic and emotion:
    response = finalChain.invoke({"topic": topic})
    st.write(response)

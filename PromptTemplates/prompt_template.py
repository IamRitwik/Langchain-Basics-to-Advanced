from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="mistral")
prompt_template = PromptTemplate(
    input_variables=["country", "noOfParas", "language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {noOfParas} short paras in {language} language
    """
)

st.title("Cuisine Info!")

country = st.text_input("Enter the country: ")
noOfParas = st.number_input("Enter the number of paras: ", min_value=1, max_value=5)
language = st.text_input("Enter the language: ")

if country:
    response = llm.invoke(prompt_template.format(country=country, noOfParas=noOfParas, language=language))
    st.write(response.content)

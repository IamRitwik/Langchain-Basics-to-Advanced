from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="mistral")
prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)

st.title("Travel Guide!")

city = st.text_input("Enter the city: ")
month = st.text_input("Enter the month: ")
language = st.text_input("Enter the language: ")
budget = st.selectbox("Travel Budget: ", ["LOW", "MEDIUM", "HIGH"])

chain = prompt_template | llm

if city and month and language and budget:
    response = chain.invoke({"city": city,
                             "month": month,
                             "language": language,
                             "budget": budget})
    st.write(response.content)

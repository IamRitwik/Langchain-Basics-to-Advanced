from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="mistral")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)
chain = prompt_template | llm
history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)
st.title("Agile Guide!")

input = st.text_input("Enter the question: ")

if input:
    response = chain_with_history.invoke({"input": input}, {"configurable": {"session_id": "session123"}})
    st.write(response.content)

st.write("History")
st.write(history_for_chain)
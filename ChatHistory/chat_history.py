from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="mistral")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)
chain = prompt_template | llm
history_for_chain = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="question",
    history_messages_key="chat_history"
)
print("Agile Guide!")

while True:
    question = input("Enter the question: ")
    if question:
        response = chain_with_history.invoke({"question": question}, {"configurable": {"session_id": "session123"}})
        print(response.content)
from langchain.chains.llm import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationSummaryMemory

# from langchain_community.chat_message_histories import FileChatMessageHistory

llm = ChatOllama(model="mistral")

# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages",
#     return_messages=True
# )

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=llm
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

while True:
    content = input(">> ")
    result = chain.invoke({"content": content})
    print(result["text"])

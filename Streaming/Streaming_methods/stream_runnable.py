from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="mistral", streaming=True)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

for chunk in chain.stream({"topic": "Tech"}):
    print(chunk, end="|", flush=True)



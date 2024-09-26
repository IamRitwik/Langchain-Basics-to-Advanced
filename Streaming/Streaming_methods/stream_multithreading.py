from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from queue import Queue
from threading import Thread


class StreamingHandler(BaseCallbackHandler):

    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, response, **kwargs):
        self.queue.put(None)


llm = ChatOllama(model="mistral", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("human", "{content}")
])

messages = prompt.format_messages(content="Tell me a joke!")


# LLMs are happy to stream, chains are unhappy to stream
# print(messages)
# for message in llm.stream(messages):
#     print(message.content)

class StreamableChain:
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            self(input, callbacks=[handler])

        Thread(target=task).start()
        while True:
            token = queue.get()
            if token is None:
                break
            yield token


class StreamingChain(StreamableChain, LLMChain):
    pass


chain = StreamingChain(llm=llm, prompt=prompt)

for output in chain.stream(input={"content": "Tell me a joke!"}):
    print(output)

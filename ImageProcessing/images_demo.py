import base64

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


llm = ChatOllama(model="mistral")
image = encode_image("airport_terminal_journey.jpeg")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images."),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm
response = chain.invoke({"input": "Explain"})

print(response.content)
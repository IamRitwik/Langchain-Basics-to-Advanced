import base64

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


llm = ChatOllama(model="mistral")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can verify identification documents."),
        (
            "human",
            [
                {"type": "text", "text": "Verify identification details"},
                {"type": "text", "text": "Name: {user_name}"},
                {"type": "text", "text": "DOB: {user_dob}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm

st.title("KYC Verification!")

uploaded_file = st.file_uploader("Upload your document: ", type=["jpg", "png", "jpeg"])

user_name = st.text_input("Enter your name: ")
user_dob = st.text_input("Enter your date of birth: ")

if uploaded_file is not None and user_dob and user_name:
    image = encode_image(uploaded_file)
    response = chain.invoke({"user_name": user_name, "user_dob": user_dob, "image": image})
    st.write(response.content)

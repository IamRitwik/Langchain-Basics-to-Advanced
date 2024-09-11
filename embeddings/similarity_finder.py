import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import numpy as np

HF_API_KEY = os.getenv("HF_API_KEY")

print(HF_API_KEY)

text1 = input("Enter the first text: ")
text2 = input("Enter the second text: ")

embeddings = HuggingFaceInferenceAPIEmbeddings(
 api_key=HF_API_KEY,
 model_name="sentence-transformers/all-MiniLM-l6-v2"
)

response1 = embeddings.embed_query(text1)
response2 = embeddings.embed_query(text2)

similarity_score = np.dot(response1, response2)

print(similarity_score*100, "%")




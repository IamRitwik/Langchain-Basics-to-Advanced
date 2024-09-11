from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
HF_API_KEY = os.getenv("HF_API_KEY")

question = input("Enter input: ")

embeddings = HuggingFaceInferenceAPIEmbeddings(
 api_key=HF_API_KEY,
 model_name="sentence-transformers/all-MiniLM-l6-v2"
)

result = embeddings.embed_query(question)
print(result)




from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
HF_API_KEY = os.getenv("HF_API_KEY")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

document = TextLoader("../RAG/job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents((document))
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever()

question = input("Enter input: ")

docs = retriever.invoke(question)

for doc in docs:
    print(doc.page_content)

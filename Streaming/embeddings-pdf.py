from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
loader = PyPDFLoader("spice.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = loader.load_and_split(
    text_splitter=text_splitter
)
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="db"
)




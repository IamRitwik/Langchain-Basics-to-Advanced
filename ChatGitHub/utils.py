from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import git
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from queue import Queue
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

allowed_extensions = ['.*','.md']


class Embedder:
    def __init__(self, git_link) -> None:
        self.retriever = None
        self.db = None
        self.num_texts = None
        self.texts = None
        self.docs = None
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.model = ChatOllama(model="mistral")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self.MyQueue = Queue(maxsize=5)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            print(dirpath)
            print(dirnames)
            print(filenames)
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as exp:
                        print("Exception occurred : " + exp)
                        pass

    def chunk_files(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_deeplake(self):
        vectorstore = Chroma.from_documents(
            self.texts,
            embedding=self.embeddings,
            persist_directory="db"
        )
        # Remove data from the cloned path
        self.delete_directory(self.clone_path)
        return vectorstore

    @staticmethod
    def delete_directory(path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def load_db(self):
        # Create and load
        self.extract_all_files()
        self.chunk_files()
        self.db = self.embed_deeplake()
        search_kwargs = {"k": 4}
        self.retriever = self.db.as_retriever(search_kwargs=search_kwargs)

    def retrieve_results(self, query):
        chat_history = list(self.MyQueue.queue)
        qa = ConversationalRetrievalChain.from_llm(self.model, chain_type="stuff", retriever=self.retriever,
                                                   condense_question_llm=self.model)
        rag_chain = qa.pick("answer")
        result = rag_chain.stream({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result))
        return result

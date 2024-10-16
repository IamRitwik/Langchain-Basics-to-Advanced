import os
import git
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

allowed_extensions = ['.*']


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

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        loader = DirectoryLoader(root_dir, glob='**/*.*', show_progress=True, loader_cls=TextLoader)
        self.docs.extend(loader.load_and_split())

    def chunk_files(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.texts = text_splitter.split_documents(self.docs)
        print(self.texts)
        self.num_texts = len(self.texts)
        print(self.num_texts)

    def load_vector_store(self):
        vectorstore = Chroma.from_documents(
            self.texts,
            embedding=self.embeddings,
            persist_directory="vector_db"
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
                for directory in dirs:
                    dir_path = os.path.join(root, directory)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def create_retriever(self):
        # Create and load
        self.extract_all_files()
        self.chunk_files()
        self.db = self.load_vector_store()
        search_kwargs = {"k": 4}
        self.retriever = self.db.as_retriever(search_kwargs=search_kwargs)

    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a coding/programming assistant and very good in searching from provided documents."
                       "Answer the user's questions based on the context: {context}."
                       "If you do not have answer from provided information, say so"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompt
        )

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human",
             "You are a coding/programming assistant. "
             "Given the above conversation and provided context, "
             "generate a search query to look up in order to get information relevant to "
             "the conversation")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=self.retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        )

        return retrieval_chain

    def process_chat(self, question, chat_history):
        chat_chain = self.create_chain()
        rag_chain = chat_chain.pick("answer")
        result = rag_chain.stream({
            "chat_history": chat_history,
            "input": question,
        })
        # response = rag_chain.invoke({
        #     "chat_history": chat_history,
        #     "input": question,
        # })
        # sources = []
        # for doc in response["context"]:
        #     sources.append(
        #         {"source": doc.metadata["source"], "page_content": doc.page_content}
        #     )
        return result

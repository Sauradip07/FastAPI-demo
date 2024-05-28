# rag_app.py
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

class RAGApp:
    def __init__(self):
        # Initialize LLM and other components
        self.llm = Ollama(model="mistral")
        self.embeddings_llm = OllamaEmbeddings(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter()

        # Load documents
        urls = [
            "https://iroha.readthedocs.io/en/develop/index.html",
            "https://iroha.readthedocs.io/en/develop/overview.html",
            "https://iroha.readthedocs.io/en/develop/concepts_architecture/index.html",
            "https://iroha.readthedocs.io/en/develop/getting_started/index.html",
            "https://iroha.readthedocs.io/en/develop/integrations/index.html",
            "https://iroha.readthedocs.io/en/develop/build/index.html",
            "https://iroha.readthedocs.io/en/develop/configure/index.html",
            "https://iroha.readthedocs.io/en/develop/deploy/index.html",
            "https://iroha.readthedocs.io/en/develop/maintenance/index.html",
            "https://iroha.readthedocs.io/en/develop/develop/index.html",
            "https://iroha.readthedocs.io/en/develop/community/index.html",
            "https://iroha.readthedocs.io/en/develop/faq/index.html",
            "https://github.com/hyperledger/iroha",
            
        ] 

        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())

        documents = self.text_splitter.split_documents(docs)
        self.vector_index = FAISS.from_documents(documents, self.embeddings_llm)
        self.retriever = self.vector_index.as_retriever()

        self.prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context and your internal knowledge.
        Give priority to context and if you are not sure then say you are not aware of topic:

        <context>
        {context}
        </context>

        Question: {input}
        """)

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def get_answer(self, question: str) -> str:
        relevant_docs = self.retriever.invoke({"input": question})
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        response = self.document_chain.invoke({
            "input": question,
            "context": [Document(page_content=context)]
        })
        
        return response

# Initialize RAGApp once
rag_app = RAGApp()

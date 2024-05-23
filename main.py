from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain


llm = Ollama(model="mistral")


loader = WebBaseLoader("https://www.anthropic.com/news/claude-3-family")
docs = loader.load()

urls = [
        "https://www.anthropic.com/news/releasing-claude-instant-1-2",
        "https://www.anthropic.com/news/claude-pro",
        "https://www.anthropic.com/news/claude-2",
        "https://www.anthropic.com/news/claude-2-1",
        "https://www.anthropic.com/news/claude-2-1-prompting",
        "https://www.anthropic.com/news/claude-3-family",
        "https://www.anthropic.com/claude"
       ] 

docs = []
for url in urls:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

embeddings_llm = OllamaEmbeddings(model="mistral")



text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)


vector_index = FAISS.from_documents(documents, embeddings_llm)

retriever = vector_index.as_retriever()

relevant_docs = retriever.invoke({"input": "Do you know about Claude 3?"})

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context and your internal knowledge.
Give priority to context and if you are not sure then say you are not aware of topic:

<context>
{context}
</context>

Question: {input}
""")

#document_chain  = prompt | llm 
document_chain = create_stuff_documents_chain(llm, prompt)

response = document_chain.invoke({
    "input": "Do you know about Claude 3?",
    "context": [Document(page_content="Claude 3 is latest Conversational AI Model from Anthropic.")]
})

print(response)

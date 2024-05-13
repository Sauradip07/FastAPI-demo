from fastapi import FastAPI
import random
import uvicorn
import nest_asyncio
#from pyngrok import ngrok
import uvicorn
# fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# import dependencies
from fastapi import FastAPI,HTTPException, Request
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import os
#import gradio as gr
# from google.colab import drive

import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import GitHubIssuesLoader ###
from langchain_community.document_loaders.merge import MergedDataLoader ###
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, dotenv_values

app = FastAPI()
# specify model huggingface mode name
model_name = "anakin87/zephyr-7b-alpha-sharded"

# function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model

# function for initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

# load model
model = load_quantized_model(model_name)

# initialize tokenizer
tokenizer = initialize_tokenizer(model_name)

# specify stop token ids
stop_token_ids = [0]

# put the url here:
url = "https://iroha.readthedocs.io/en/develop/index.html"

# load documents from urls
loader_web = RecursiveUrlLoader(url=url)

# set the github repo and remember to add your github token to secret keys
# from google.colab import userdata
# GPAT = userdata.get('GITHUB_PERSONAL_ACCESS_TOKEN')
load_dotenv()
GPAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

loader_github = GitHubIssuesLoader(
    repo="hyperledger/iroha",
    access_token=GPAT,
)

loader= MergedDataLoader(loaders=[loader_web, loader_github])

documents = loader.load()

# filter metadata types fixing an exeption
documents = filter_complex_metadata(documents, allowed_types=(str, int, float, bool))

# split the documents in small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #Chage the chunk_size and chunk_overlap as needed
all_splits = text_splitter.split_documents(documents)

# specify embedding model (using huggingface sentence transformer)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

#embed document chunks
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# specify the retriever
retriever = vectordb.as_retriever()

# build huggingface pipeline for using zephyr-7b-alpha
pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

# specify the llm
llm = HuggingFacePipeline(pipeline=pipeline)

# build conversational retrieval chain with memory (rag) using langchain
def create_conversation(query: str, chat_history: list) -> tuple:
    try:

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=False
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )

        result = qa_chain({'question': query, 'chat_history': chat_history})
        chat_history.append((query, result['answer']))
        return '', chat_history


    except Exception as e:
        chat_history.append((query, e))
        return '', chat_history

class Query(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "LLM Chatbot is Up and Running!"}


@app.post("/chat/")
async def chat(query: Query):
    chat_history = []
    _, chat_history = create_conversation(query.query, chat_history)
    return {"chat_history": chat_history}



# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
# uvicorn.run(app, port=8000)
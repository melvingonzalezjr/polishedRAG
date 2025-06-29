import chainlit as cl

import chromadb
from chromadb.config import Settings

from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.embeddings import Embeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

########## PROMPT TEMPLATE ###########
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful restaurant assistant for the servers and bartenders at Pinstripes. You will answer questions about the menu and drinks based" +
        "on the information provided in the chunks given to you."),
        ("human", "Hello polishedRAG! I am a server here at Pinstripes and need help with a customer question."),
        ("ai", "Certainly! I can help you with that. What is your question?"),
        ("human", "{question}")
    ]
)

########## LOAD THE PDF FILES ###########
food_loader = PyPDFLoader(file_path="data/food_menu.pdf")
food_data = food_loader.load()

wine_loader = PyPDFLoader(file_path="data/wine_menu.pdf")
wine_data = wine_loader.load()

########## SPLIT THE DOCUMENTS INTO CHUNKS ###########
# Split the documents into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

# Split the food and wine documents into smaller chunks and store them into a single list
food_and_wine_docs = text_splitter.split_documents(food_data + wine_data)
print(len(food_and_wine_docs), food_and_wine_docs[3])


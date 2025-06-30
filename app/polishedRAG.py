import chainlit as cl

import chromadb
from chromadb.config import Settings
from langchain import hub
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# ########## PROMPT TEMPLATE ###########
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful restaurant assistant for the servers and bartenders at Pinstripes. You will answer questions about the menu and drinks based" +
#         "on the information provided in the chunks given to you."),
#         ("human", "Hello polishedRAG! I am a server here at Pinstripes and need help with a customer question."),
#         ("ai", "Certainly! I can help you with that. What is your question?"),
#         ("human", "{question}")
#     ]
# )

########## LOAD THE PDF FILES ###########
food_loader = PyPDFLoader(file_path="data/food_menu.pdf")
food_data = food_loader.load()

wine_loader = PyPDFLoader(file_path="data/wine_menu.pdf")
wine_data = wine_loader.load()

########## SPLIT THE DOCUMENTS INTO CHUNKS ###########
# Create the text splitter for chunking the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True    #track index. Needed for Chroma
)

# Split the food and wine documents into smaller chunks and store them into a single list
food_and_wine_docs = text_splitter.split_documents(food_data + wine_data)

#create a single list of chunks (w/o metadata)
# food_wine_document = []
# for doc in food_and_wine_docs:
#     chunk = doc.page_content
#     food_wine_document.append(chunk)

# ########## EMBEDDINGS ###########
# #Create the embeddings model (using free local model)
# embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = Chroma(
    collection_name="pinstripes_menu_collection",
    embedding_function=embeddings,
)

pinstripes_document_ids = vector_store.add_documents(documents=food_and_wine_docs)

prompt = hub.pull("rlm/rag-prompt")

question = "Are the ribs made from pork or beef?"

retrieved_docs = vector_store.similarity_search(question)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
prompt = prompt.invoke({"question": question, "context": docs_content})

# Using Ollama with Llama 3.2 (free local model)
# Make sure to install Ollama and pull the model first:
# ollama pull llama3.2:3b
llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
answer = llm.invoke(prompt)

print(answer)

# #Create the embeddings vector
# embeddings = embeddings_model.embed_documents(food_wine_document)

# ########## VECTOR STORE ###########
# #Create Choma client
# chroma_client = chromadb.Client()

# #Create collection
# collection = chroma_client.create_collection(name="pinstripes_menu_collection")

# #Add text documents to the collection
# ids = [f"doc_{i}" for i in range(len(food_wine_document))]

# collection.add(
#     documents=food_wine_document,
#     ids=ids,
#     embeddings=embeddings
# )

# #test results
# results = collection.query(
#     query_texts=["Are the ribs made from pork or beef?"],
#     n_results=1
# )

# print("Results:", results)






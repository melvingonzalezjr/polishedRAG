from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the food and wine menus
food_loader = PyPDFLoader(file_path="data/food_menu.pdf")
food_data = food_loader.load()

wine_loader = PyPDFLoader(file_path="data/wine_menu.pdf")
wine_data = wine_loader.load()

print(f"Loaded {len(food_data)} food documents and {len(wine_data)} wine documents.")

# Split the documents into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

# Split the food and wine documents into smaller chunks and store them into a single list
food_and_wine_docs = text_splitter.split_documents(food_data + wine_data)

print(f"Split into {len(food_and_wine_docs)} chunks.")

@cl.on_chat_start
async def on_chat_start():

    # Start with a chat model 
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        streaming = True
    )

    # Create a chat prompt template that will be used to format the input for the model
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful restaurant assistant for the servers and bartenders at Pinstripes. You will answer questions about the menu and drinks based" +
            "on the information provided in the menu and drinks documents stored in ../data. You may also answer questions based on what you know or learn about food and drinks" +
            ", but all responses should directly relate to the menu and drinks at Pinstripes."),

            ("human", "{question}")
        ]
    )

    # Create an LLMChain that combines the model and the prompt
    # The output will be a string, so we use StrOutputParser to parse the output
    # A chainis one or more series of LLM calls that can be used to process input and generate output
    chain= LLMChain(
        llm=model,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    # Set the chain in the user session so it can be used later
    # This allows us to access the chain in the main function when a message is received
    cl.user_session.set("chain", chain)


########## This is the main function that will be called when a message is received #############

# Proper decorator to main function so Chainlit can handle the function when it receives a message
# async just means that this function can run asynchronously, allowing Chainlit to handle multiple requests efficiently. Equivalent to async in JS
# this is our main function that will be called after first awaiting for message to be sent by user
@cl.on_message
async def main(message: cl.Message):

    # loading the chain from the user session
    chain = cl.user_session.get("chain")
    
    response = await chain.arun(
        question=message.content,
        callbacks=[cl.LangchainCallbackHandler()]  # This will handle the streaming response and send it back to the user

    )

    await cl.Message(content=response).send()


from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
import chainlit as cl

# # Start with a chat model 
# model = ChatOpenAI(
#     model="gpt-4.1-nano",
#     streaming = True
# )

# # Create a chat prompt template that will be used to format the input for the model
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful restaurant assistant for the servers and bartenders at Pinstripes. You will answer questions about the menu and drinks based" +
#         "on the information provided in the menu and drinks documents stored in data. You may also answer questions based on what you know or learn about food and drinks" +
#         ", but all responses should directly relate to the menu and drinks at Pinstripes."),

#         ("human", "{question}")
#     ]
# )

# # Create an LLMChain that combines the model and the prompt
# # The output will be a string, so we use StrOutputParser to parse the output
# chain= LLMChain(
#     llm=model,
#     prompt=prompt,
#     output_parser=StrOutputParser()
# )

# Proper decorator to main function so Chainlit can handle the function when it receives a message


# async just means that this function can run asynchronously, allowing Chainlit to handle multiple requests efficiently. Equivalent to async in JS
# this is our main function that will be called after first awaiting for message to be sent by user
@cl.on_message
async def main(message: cl.Message):
    await cl.Message(message.content).send()


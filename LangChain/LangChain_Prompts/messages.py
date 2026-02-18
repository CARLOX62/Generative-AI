from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import  HuggingFacePipeline, ChatHuggingFace
# from dotenv import load_dotenv
# import streamlit as st


# load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        return_full_text = False,
    )
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assistent'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
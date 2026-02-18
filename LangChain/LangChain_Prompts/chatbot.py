from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_core.prompts import PromptTemplate, load_prompt
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

# for memory
chat_history = [
    SystemMessage(content='you are a helpful AI assistent')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    formatted_prompt = f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
    result = model.invoke(chat_history, config={"stop": ["</s>", "<|user|>"]})
    chat_history.append(AIMessage(content = result.content))
    print("AI: ",result.content.strip())

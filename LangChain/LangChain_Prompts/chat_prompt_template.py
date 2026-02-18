from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    # SystemMessage(content='You are a helpful {domian} expert'),
    # HumanMessage(content='Explain in simple terms, waht is {topic}')
    ('system', 'You are a helpful {domian} expert'),
    ('human', 'Explain in simple terms, waht is {topic}')
])

prompt = chat_template.invoke({'domian':'cricket','topic':'Dusra'})

print(prompt)
import streamlit as st
from LangGraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# st.session_state is a dictionary-like object that allows you to store and retrieve values across multiple runs of your Streamlit app. It is useful for maintaining state in interactive applications.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# Display the chat messages from history on app rerun
for message in st.session_state['message_history']:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.text(message['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type your message here...')

if user_input:
    # Add user message to the chat history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Get the assistant's response from the chatbot
    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= {'configurable': {'thread_id': 'thread-1'}},
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})    
import streamlit as st
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# Access the API key from the secrets configuration
config = st.secrets["api_keys"]
openai_api_key = config["openai_api_key"]

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Initialize the session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Define the prompt template
template = """
You are a health assistant. Continue the following conversation and answer the next question:

{history}

User: {query}
Assistant:
"""
prompt = PromptTemplate(template=template, input_variables=["history", "query"])
llm_chain = prompt | llm

# Streamlit app layout
st.title("Health Chatbot")

# Chat display
for message in st.session_state['conversation_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your health-related question:"):
    # User message
    st.session_state['conversation_history'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Thinking..."):
        history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['conversation_history'] if msg['role'] == "user"])
        response = llm_chain.invoke({"history": history, "query": prompt}).content.strip()

        # Assistant message
        st.session_state['conversation_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Option to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state['conversation_history'] = []
    st.success("Conversation history cleared.")

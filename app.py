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

# Display the conversation history in a chat-like format
for i, message in enumerate(st.session_state['conversation_history']):
    if i % 2 == 0:
        st.chat_message("user", message)
    else:
        st.chat_message("assistant", message)

# User input
query = st.text_input("Ask your health-related question:", key="input")

# Generate response when the user submits a question
if st.button("Send", key="send"):
    if query:
        # Append the user's message to the conversation history
        st.session_state['conversation_history'].append(f"User: {query}")
        
        with st.spinner("Assistant is typing..."):
            # Construct the conversation history for the prompt
            history = "\n".join(st.session_state['conversation_history'])

            # Invoke the LLM with the conversation history
            response = llm_chain.invoke({"history": history, "query": query}).content.strip()

            # Append the assistant's response to the conversation history
            st.session_state['conversation_history'].append(f"Assistant: {response}")

            # Refresh the page to display the new messages in the chat format
            st.experimental_rerun()
    else:
        st.warning("Please enter a question.")

# Option to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state['conversation_history'] = []
    st.experimental_rerun()


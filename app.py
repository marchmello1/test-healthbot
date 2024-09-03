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

# User input
query = st.text_input("Ask your health-related question:")

# Generate response when the user submits a question
if st.button("Submit"):
    if query:
        with st.spinner("Thinking..."):
            # Construct the conversation history
            history = "\n".join(st.session_state['conversation_history'])

            # Invoke the LLM with the conversation history
            response = llm_chain.invoke({"history": history, "query": query}).content.strip()

            # Update the conversation history
            st.session_state['conversation_history'].append(f"User: {query}")
            st.session_state['conversation_history'].append(f"Assistant: {response}")

            # Display the response
            st.success("Health Assistant's response:")
            st.write(response)
    else:
        st.warning("Please enter a question.")

# Option to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state['conversation_history'] = []
    st.success("Conversation history cleared.")

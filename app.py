import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI  # Updated to import from langchain_community
from langchain.prompts import PromptTemplate
import openai

# Access the OpenAI API key securely from Streamlit secrets
config = st.secrets["api_keys"]
openai_api_key = config["openai_api_key"]

# Initialize OpenAI API key
openai.api_key = openai_api_key

# Define a custom prompt for the chatbot
template = """
You are a knowledgeable health assistant. When provided with symptoms or a health-related question, respond with a likely diagnosis, causes, and treatment recommendations.

User: {user_input}
Assistant:
"""

# Create a PromptTemplate and LLMChain
prompt = PromptTemplate(input_variables=["user_input"], template=template)
llm_chain = LLMChain(
    llm=OpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key=openai_api_key),  # Pass the API key here
    prompt_template=prompt
)

# Streamlit App
st.title("Health Chatbot")
st.write("Ask any health-related question or describe your symptoms to get advice.")

# User input
user_input = st.text_input("Enter your symptoms or question:")

if user_input:
    # Get the response from GPT-3.5
    response = llm_chain.run({"user_input": user_input})
    
    # Display the response
    st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    st._run()

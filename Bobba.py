import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import requests
from sklearn.ensemble import RandomForestRegressor
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load data from URL
@st.cache_data
def load_data_from_url():
    """Load data from the given URL."""
    file_url = 'https://github.com/PatricRc/BobbaLab/blob/main/BobbaSales.xlsx?raw=true'
    try:
        response = requests.get(file_url, timeout=30)
        if response.status_code == 200:
            file_data = io.BytesIO(response.content)
            df = pd.read_excel(file_data, engine='openpyxl')
            return df
        else:
            st.error(f"HTTP request failed with status code: {response.status_code}")
            st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading the dataset: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()

# Process user queries
# Chat with data using OpenAI
def chat_with_data(df, input_text, openai_api_key):
    """Chat with the survey data using OpenAI."""
    try:
        # Use the processed query for a summary response
        context = df.to_string(index=False)[:10000]  # Limit to avoid API constraints

        # Create a prompt template
        message = f"""
        Context:
        {context}

        Question:
        {input_text}

        Answer:
        """

        # Initialize OpenAI LLM with model 'gpt-4o-2024-08-06'
        if not openai_api_key:
            st.warning("Please enter a valid OpenAI API key.")
            return
        llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", openai_api_key=openai_api_key)

        # Generate response
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

# Main function
def main():
    # API key input
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if not openai_api_key:
        st.warning("Please enter a valid OpenAI API key.")
        return
    st.title("Chat with Your CSV/XLSX Data")

    # Load data
    df = load_data_from_url()
    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df)

        # User query input
        input_text = st.text_area("Ask a question about your data:")

        # Chat button
        if st.button("Chat with Data") and input_text:
            chat_with_data(df, input_text, openai_api_key)

if __name__ == "__main__":
    main()

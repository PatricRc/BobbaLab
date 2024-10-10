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

@st.cache_data
def load_data_from_url():
    """Load data from the given URL."""
    # Load the dataset from the GitHub repository
    file_url = 'https://github.com/PatricRc/BobbaLab/blob/main/BobbaSales.xlsx?raw=true'
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise an error for bad status codes
        file_data = io.BytesIO(response.content)
        df = pd.read_excel(file_data, engine='openpyxl', sheet_name='Sheet1')
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading the dataset: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()

def chat_with_data(df_chat, input_text, api_key):
    """Chat with the survey data using OpenAI."""
    try:
        # Convert DataFrame to a format suitable for context
        context = df_chat.to_string(index=False)

        # Create a prompt template
        message = f"""
        Answer the following question using the context provided:

        Context:
        {context}

        Question:
        {input_text}

        Answer:
        """

        # Initialize OpenAI LLM with model 'gpt-3.5-turbo'
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

        # Generate response
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

def main():
    st.title("Chat with Your CSV/XLSX Data")

    # API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Load data
    df = load_data_from_url()
    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df)

        # User query input
        input_text = st.text_area("Ask a question about your data:")

        # Chat button
        if st.button("Chat with Data") and input_text:
            chat_with_data(df, input_text, api_key)

if __name__ == "__main__":
    main()

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

# Temporarily disable caching for debugging
# @st.cache_data
def load_data_from_url():
    """Load data from the given URL."""
    # Load the dataset from the GitHub repository
                    file_url = 'https://github.com/PatricRc/BobbaLab/blob/main/BobbaSales.xlsx?raw=true'
                    try:
        response = requests.get(file_url, timeout=30)
                                        if response.status_code == 200:
            file_data = io.BytesIO(response.content)
                                                xlsx = pd.ExcelFile(file_data, engine='openpyxl')
            df = pd.read_excel(xlsx, sheet_name=xlsx.sheet_names[0])  # Load the first sheet as default
            return df
        else:
                                                            st.error(f"HTTP request failed with status code: {response.status_code}")
            st.stop()  # Raise an error for bad status codes
        file_data = io.BytesIO(response.content)
                xlsx = pd.ExcelFile(file_data, engine='openpyxl')
                df = pd.read_excel(xlsx, sheet_name=xlsx.sheet_names[0])  # Load the first sheet as default
                return df
                    except requests.exceptions.RequestException as e:
        st.error(f"Error loading the dataset: {e}")
        st.stop()
                    except ValueError as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()

def chat_with_data(df_chat, input_text, openai_api_key):
    """Chat with the survey data using OpenAI."""
    try:
        # Convert DataFrame to a more manageable format
        max_context_length = 3000  # Limit the context length to avoid exceeding API limits
        filtered_df = df_chat[df_chat.apply(lambda row: input_text.split()[-1] in row.values.astype(str), axis=1)]
        if not filtered_df.empty:
            context = filtered_df.to_string(index=False)
        else:
            context = df_chat.to_string(index=False)  # Use a general context if no specific filter is found  # Use only the first 100 rows for context

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
        
        if not openai_api_key:
            st.warning("Please enter a valid OpenAI API key.")
            return
        llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", openai_api_key=openai_api_key)

        # Generate response
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

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

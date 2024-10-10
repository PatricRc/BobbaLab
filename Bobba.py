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
    st.write("Attempting to load data from URL...")
    """Load data from the given URL."""
    # Load the dataset from the GitHub repository
    file_url = 'https://github.com/PatricRc/BobbaLab/blob/main/BobbaSales.xlsx?raw=true'
    try:
        response = requests.get(file_url, timeout=30)
        if response.status_code == 200:
            st.write("HTTP request successful.")
        else:
            st.error(f"HTTP request failed with status code: {response.status_code}")
            st.stop()  # Raise an error for bad status codes
        file_data = io.BytesIO(response.content)
        st.write("Data successfully fetched from URL.")
        xlsx = pd.ExcelFile(file_data, engine='openpyxl')
        st.write("Available sheets:", xlsx.sheet_names)
        df = pd.read_excel(xlsx, sheet_name=xlsx.sheet_names[0])  # Load the first sheet as default
        st.write(f"Excel file successfully read. Loaded sheet: {xlsx.sheet_names[0]}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading the dataset: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()

def chat_with_data(df_chat, input_text):
    """Chat with the survey data using OpenAI."""
    try:
        # Convert DataFrame to a more manageable format
        max_context_length = 3000  # Limit the context length to avoid exceeding API limits
        filtered_df = df_chat[df_chat.apply(lambda row: input_text.split()[-1] in row.values.astype(str), axis=1)]
        if not filtered_df.empty:
            context = filtered_df.to_string(index=False)
        else:
            context = df_chat.head(10).to_string(index=False)  # Use a general context if no specific filter is found  # Use only the first 100 rows for context

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
        llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", openai_api_key="sk-RbwIZK5Qb1b_sWhMug-YhDNOmCwNrGcJ11OQbbdkuFT3BlbkFJdN1-K8s7iabMUeEeHMvDMjekBpcmjworHQyPQEYlAA")

        # Generate response
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

def main():
    st.title("Chat with Your CSV/XLSX Data")

    # API key input
        # Load data
    df = load_data_from_url()
    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df)

        # User query input
        input_text = st.text_area("Ask a question about your data:")

        # Chat button
        if st.button("Chat with Data") and input_text:
            chat_with_data(df, input_text)

if __name__ == "__main__":
    main()

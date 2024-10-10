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
def load_data(uploaded_file):
    """Load data from the uploaded file."""
    try:
        # Load based on file type
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return None

        return df

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None

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

    # File uploader
    uploaded_file = st.file_uploader("Upload an XLSX or CSV file", type=["xlsx", "csv"])

    # API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file)
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

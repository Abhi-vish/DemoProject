import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import json

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
hub_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def preprocess_dataset(data):
    df = data.copy()
    df.fillna({'category': 'Unknown', 'SubCategory': 'Unknown', 'ProductName': '', 'Description': ''}, inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    return df

def get_data_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(data_chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(data_chunks, embedding)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="qwen2:1.5b", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_question(user_question):
    response = None
    if st.session_state.conversation is not None:
        try:
            # Get the response from the conversation chain
            response = st.session_state.conversation({'question': user_question})

            # Extract question and answer
            question = response.get('question', '')
            answer = response.get('answer', '')

            # Display the question and answer
            st.write(f"Question: {question}")
            st.write(f"Answer: {answer}")

            print(response)                      

        except Exception as e:
            st.write(f"An unexpected error occurred: {e}")
            st.write("Response from model:", response)
    else:
        st.write("Please process the data first by clicking the 'Process' button.")

def main():
    dataset_path = "Data/Luxury_Products_Apparel_Data.csv"
    data = pd.read_csv(dataset_path)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Query With Dataset')
    user_question = st.chat_input('Ask a question related to the dataset')
    if user_question:
        with st.spinner('Generating response...'):
            handle_user_question(user_question)

    with st.sidebar:
        st.write("## Dataset")
        st.write(data.head(n=5))
        if st.button("Process"):
            if not data.empty:
                with st.spinner('Processing...'):
                    preprocessed_data = preprocess_dataset(data)

                    text_chunks = get_data_chunks(' '.join(preprocessed_data['Description'].astype(str)))

                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.write('Data processed successfully')
                    st.write(preprocessed_data.head(n=5))
                    st.write('Data Chunked Successfully')
                    st.write(text_chunks)
            else:
                st.write('No data found')

if __name__ == '__main__':
    main()

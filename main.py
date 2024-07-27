# Required Libraries
import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

class DatasetAnalyzer:
    def __init__(self, data):
        """
        Initialize the DatasetAnalyzer class with the given data.
        
        Args:
            data (pd.DataFrame): The dataset to be analyzed.
        """
        self.data = data

    def config(self):
        """
        Configure the Streamlit page settings.
        
        This function sets the page title, icon, layout, sidebar state,
        and menu items for the Streamlit app.
        """
        st.set_page_config(
            page_title="Dataset Analyzer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.streamlit.io/',
            }
        )

    def preprocess_data(self, data):
        """
        Preprocess the dataset by dropping unnecessary columns, filling missing values,
        and removing duplicate rows.
        
        Args:
            data (pd.DataFrame): The original dataset.
            
        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        # Drop unnecessary columns
        preprocessed_data = data.drop(columns=['Unnamed: 0', 'Category', 'SubCategory'], axis=1)
        # Fill missing values
        preprocessed_data.fillna({'category': 'Unknown', 'SubCategory': 'Unknown', 'ProductName': '', 'Description': ''}, inplace=True)
        # Remove duplicate rows
        preprocessed_data = preprocessed_data.drop_duplicates()
        return preprocessed_data
    
    def get_data_chunks(self, text):
        """
        Split the text data into smaller chunks.
        
        Args:
            text (str): The text to be split into chunks.
            
        Returns:
            list: A list of text chunks.
        """
        text_splitter = CharacterTextSplitter(
            separator=",",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vectorstore(self, data_chunks):
        """
        Create a vector store from the data chunks using HuggingFace embeddings.
        
        Args:
            data_chunks (list): A list of text chunks.
            
        Returns:
            FAISS: The vector store created from the data chunks.
        """
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(data_chunks, embedding)
        return vectorstore
    
    def conversation_chain(self, vectorstore):
        """
        Create a conversational retrieval chain using the given vector store.
        
        Args:
            vectorstore (FAISS): The vector store for retrieval.
            
        Returns:
            ConversationalRetrievalChain: The conversational retrieval chain.
        """
        llm = ChatOllama(model="qwen2:1.5b", temperature=0)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    
    def handle_user_question(self, user_question):
        """
        Handle the user's question and generate a response using the conversational chain.
        
        Args:
            user_question (str): The user's question.
            
        Returns:
            str: The generated response.
        """
        response = None
        if st.session_state.conversation is not None:
            response = st.session_state.conversation(user_question)
        return response

    def run(self, data):
        """
        Run the data preprocessing, chunking, and vector store creation processes.
        
        Args:
            data (pd.DataFrame): The original dataset.
            
        Returns:
            ConversationalRetrievalChain: The conversational retrieval chain created from the dataset.
        """
        preprocessed_data = self.preprocess_data(data)
        st.write('Data:', preprocessed_data.head())
        data_chunks = self.get_data_chunks(preprocessed_data[['ProductName', 'Description']].astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=','))
        st.write('Data chunks:', len(data_chunks))
        vectorstore = self.get_vectorstore(data_chunks)
        st.write('Vectorstore:', vectorstore)
        conversation_chain = self.conversation_chain(vectorstore)
        return conversation_chain
    
    def user_interface(self):
        """
        Create the user interface for the Streamlit app.
        
        This function sets up the page configuration, initializes session states,
        handles user input, and displays the dataset and response.
        """
        self.config()
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.header('Query With Dataset')
        st.divider()
        user_question = st.chat_input('Ask a question related to the dataset')
        if user_question:
            with st.spinner('Generating response...'):
                response = self.handle_user_question(user_question)
                if response:
                    st.write('Response:', response)
                else:
                    st.write('No response generated.')

        with st.sidebar:
            st.write("## Dataset")
            st.write(self.data.head(n=5))
            if st.button("Process"):
                if not self.data.empty:
                    with st.spinner('Processing...'):
                        conversation_chain = self.run(self.data)
                        st.session_state.conversation = conversation_chain
                else:
                    st.write('No data found')


# To use the class
# data = pd.read_csv('your_dataset.csv')  # Load your dataset here
# analyzer = DatasetAnalyzer(data)
# analyzer.user_interface()
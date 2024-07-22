
## Overview

This project is a Streamlit application designed to interact with a dataset and answer user queries based on the dataset's content. The application uses LangChain to handle conversational queries, process dataset descriptions, and return relevant responses. It also utilizes Hugging Face and FAISS for embeddings and vector store operations, along with Ollama's `qwen2:1.5b` LLM model for generating responses.

## Features

- **Dataset Processing**: Preprocesses and chunks text data from a dataset.
- **Conversational Queries**: Allows users to ask questions related to the dataset and retrieves answers using a conversational model.
- **Chat History**: Displays the chat history including the questions and answers.

## Technologies Used

- **Streamlit**: For building the web interface.
- **Pandas**: For data manipulation and preprocessing.
- **LangChain**: For handling conversational queries and integrating memory.
- **Hugging Face Embeddings**: For generating embeddings for the text data.
- **FAISS**: For efficient similarity search and vector storage.
- **Ollama**: Utilizes the `qwen2:1.5b` LLM model for generating responses.
- **dotenv**: For managing environment variables.
## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Abhi-vish/DemoProject
   cd DemoProject
   ```
2. **Requiremennts.txt**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run**
   ```bash
   streamlit run app.py
   ```


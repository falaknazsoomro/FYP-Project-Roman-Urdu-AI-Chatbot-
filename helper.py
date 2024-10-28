from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader  # Updated import
import os

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings


def load_and_concatenate_files(file_paths=['RomanUrduDataSet.csv']):
    all_data = ""
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_data += file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return all_data


# Function to split text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20
    )
    text_chunks = text_splitter.split_text(extracted_data)
    return text_chunks

#Embedding Model
def OpenAIEmbeddings_Model():
   embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
   return embeddings

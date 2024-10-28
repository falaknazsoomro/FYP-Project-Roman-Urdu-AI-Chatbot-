from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain.vectorstores import Chroma
from src.helper import load_and_concatenate_files, text_split, OpenAIEmbeddings_Model

# Load environment variables
load_dotenv()

# Ensure that the OpenAI API key is set in the environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

extracted_data = load_and_concatenate_files()
text_chunks = text_split(extracted_data)
embeddings = OpenAIEmbeddings_Model()

from langchain.schema import Document

# Convert texts to Document objects
documents = [Document(page_content=t) for t in text_chunks]

# Initialize vector database
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory='db'
)

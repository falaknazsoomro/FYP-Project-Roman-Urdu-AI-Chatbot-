from flask import Flask, render_template, request
from src.helper import OpenAIEmbeddings_Model
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize the Gemini Pro LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

embeddings = OpenAIEmbeddings_Model()

vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})


# Ensure prompt_template is a string
prompt =prompt_template

# Initialize PromptTemplate correctly
PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('index.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_emebddings
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load the environment variable
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Download Embeddings Model from hugging face
embeddings = download_hugging_face_emebddings()

# Correct usage of environment variable
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  # Note: key must be passed as a string

# Set the correct Pinecone index name
index_name = "medical-chatbot"

# Loading the exising index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

question_answer = RetrievalQA.from_chain_type(llm=llm,
                                              chain_type="stuff",
                                              retriever=docsearch.as_retriever(search_kwargs={'k':2}),
                                              return_source_documents=True,
                                              chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg # ✅ safe name
    print("User input:", user_input)
    
    result = question_answer({"query": user_input})
    print("Reponse:", result["result"])
    
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
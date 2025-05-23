from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.helper import load_pdf, text_split, download_hugging_face_emebddings
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Load the PDF
extract_data = load_pdf("data/")

# Get the Chunks
text_chunks = text_split(extract_data)

# Download Embeddings Model from hugging face
embeddings = download_hugging_face_emebddings()

# Correct usage of environment variable
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  # Note: key must be passed as a string

# Set the correct Pinecone index name
index_name = "medical-chatbot"

# Make sure index exists
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Use LangChain to store embeddings in Pinecone
docsearch = PineconeVectorStore.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name  # ✅ Use index_name (string), not the Pinecone object
)
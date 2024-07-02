from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import PGVector
from langchain_postgres import PGVector
import json
from pathlib import Path
from pprint import pprint

connection_string = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
collection_name = "ai_search_vector"


file_path='master-data 1.json'
loader = JSONLoader(
    file_path= file_path,
    jq_schema='.[]',
    text_content=False)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=[".\n",". ","\n.","\n\n"],
    chunk_size=512,
    chunk_overlap=128,
     keep_separator= False,
    length_function=len,
)

docs = text_splitter.split_documents(data)

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

db = PGVector.from_documents(embedding=embeddings,documents=docs,
              collection_name=collection_name,connection_string=connection_string)

print("PG Vector Embeddings Sucessfull")


  

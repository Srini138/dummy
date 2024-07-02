from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_postgres import PGVector
from langchain_community.vectorstores import PGVector

connection_string = "postgresql+psycopg2://postgres:srini@localhost:5432/postgres"
collection_name = "ai_search_vector"

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

vectorstore = PGVector(
    embedding_function=embeddings,
    collection_name=collection_name,
    connection_string=connection_string,
    use_jsonb=True,
)

query = "who is ceo"
similarity = vectorstore.similarity_search_with_score(query=query)

for i in similarity:
    print(i)
    print('------------------------------')

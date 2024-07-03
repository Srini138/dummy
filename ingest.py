from langchain_community.document_loaders import JSONLoader
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_postgres import PGVector
import json
from pathlib import Path
from pprint import pprint



def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["selectorName"] = record.get("selectorName")
    metadata["software"] = record.get("software")
    metadata["country"] = record.get("country")
    metadata["language"] = record.get("language")
    return metadata


def main(model_name):

    connection_string = f"postgresql+psycopg2://postgres:srini@localhost:5432/{model_name}"
    collection_name = f"ai_search_{model_name}_vector"

    model_mapping = {
        'allminilm': 'sentence-transformers/all-MiniLM-l6-v2',
        'instructor_large': 'hkunlp/instructor-large',
        'e5_large': 'intfloat/e5-large-v2',
        'e5_base': 'intfloat/e5-base-v2',
        'instructor_xl': 'hkunlp/instructor-xl',

    }
    if model_name not in model_mapping:
        raise ValueError(f"Model name '{model_name}' not recognized. Available options: {list(model_mapping.keys())}")
    
    file_path='master-data 1.json'
    loader = JSONLoader(
        file_path= file_path,
        jq_schema='.[]',
        metadata_func=metadata_func,
        text_content=False)

    data = loader.load()

    # Define the path to the pre-trained model you want to use
    model_path = model_mapping[model_name]

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    encode_kwargs = {'normalize_embeddings': True}

    if 'instructor' in model_name:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    db = PGVector.from_documents(embedding=embeddings,documents=data,
                collection_name=collection_name,connection=connection_string)

    print(f"PG Vector Embeddings Sucessfull for model {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest embeddings into PostgreSQL with pgvector")
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to use for embeddings')
    args = parser.parse_args()
    main(args.model_name)
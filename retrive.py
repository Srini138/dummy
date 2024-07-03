from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import time
from langchain_postgres import PGVector
import json

# query = "outdoor wall temp sensor"

# Define a list of models and their corresponding collection names
models = [
    {
        "model_name": "sentence-transformers/all-MiniLM-l6-v2",
        "collection_name": "ai_search_allminilm_vector",
        "connection_name":"allminilm"
    },
    {
        "model_name": "hkunlp/instructor-large",
        "collection_name": "ai_search_instructor_large_vector",
        "connection_name":"instructor_large"
    },
    {
        "model_name": "hkunlp/instructor-xl",
        "collection_name": "ai_search_instructor_xl_vector",
        "connection_name":"instructor_xl"
    },
    {
        "model_name": "intfloat/e5-large-v2",
        "collection_name": "ai_search_e5_large_vector",
        "connection_name":"e5_large"
    },
    {
        "model_name": "intfloat/e5-base-v2",
        "collection_name": "ai_search_e5_base_vector",
        "connection_name":"e5_base"
    }
]

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to True
encode_kwargs = {'normalize_embeddings': True}


def main(query):
    # Initialize a dictionary to store the results
    results = {}

    # Iterate over each model and perform the similarity search
    for model_info in models:
        model_path = model_info["model_name"]
        collection_name = model_info["collection_name"]
        connect = model_info["connection_name"]
        connection_string = f"postgresql+psycopg2://postgres:srini@localhost:5432/{connect}"
        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        if 'instructor' in model_path:
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

        # Initialize the PGVector instance for the current collection
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )

        start_time = time.time()
        # Perform the similarity search
        similarity = vectorstore.similarity_search_with_score(query=query, k=2)
        end_time = time.time()
        time_taken = end_time - start_time

        # Extract and format the results and scores
        formatted_results = [{"response": json.loads(item[0].page_content), "score": item[1]} for item in similarity]

        # Save the results in the dictionary
        results[collection_name] = {
            "similarity_results": formatted_results,
            "time_taken": time_taken
        }

    return results

    # # Print the results
    # for collection_name, data in results.items():
    #     print(f"Results from collection: {collection_name}")
    #     print(f"Time taken: {data['time_taken']} seconds")
    #     for result in data["similarity_results"]:
    #         pprint(result)
    #         print('------------------------------')


# if __name__ == "__main__":
#     query = "outdoor wall temp sensor"
#     main(query)
    












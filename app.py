from fastapi import FastAPI, Query
from pydantic import BaseModel
from retrive import main
# Define the FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search(request: QueryRequest):
    query = request.query
    json_response = main(query)
    return json_response



# start the application
# uvicorn app:app --reload


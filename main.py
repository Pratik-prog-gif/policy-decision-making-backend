from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from model_handler import load_and_process_document, get_relevant_chunks
from decision_chain import get_decision_chain
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
vectorstore = None

class QueryInput(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile):
    global vectorstore
    try:
        vectorstore = await load_and_process_document(file)
        return {"message": "Vectorstore created successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query")
async def query_doc(input: QueryInput):
    global vectorstore
    if vectorstore is None:
        return JSONResponse(status_code=400, content={"error": "No document uploaded yet."})

    try:
        docs = get_relevant_chunks(vectorstore, input.question)
        context = "\n\n".join(doc.page_content[:1000] for doc in docs)

        chain = get_decision_chain()
        response = chain.invoke({
            "question": input.question,
            "context": context
        })

        return {"response": response.content, "matched_clauses": [doc.page_content for doc in docs]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

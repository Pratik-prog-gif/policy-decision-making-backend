import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

async def load_and_process_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await uploaded_file.read()
        tmp_file.write(content)
        temp_path = tmp_file.name

   
    if suffix == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(temp_path)
    elif suffix in [".eml", ".msg"]:
        loader = UnstructuredEmailLoader(temp_path)
    else:
        raise ValueError("Unsupported file format.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def get_relevant_chunks(vectorstore, query, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import asyncio  # Import the asyncio library

# --- LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- FIX for asyncio error on Render/Windows ---
# This manually creates and sets the event loop for the current thread,
# which is required by some LangChain components in certain server environments.
asyncio.set_event_loop(asyncio.new_event_loop())

# --- App Configuration ---
app = FastAPI(
    title="Career Recommendation AI API",
    description="An API for getting career recommendations based on a knowledge base.",
    version="1.0.0"
)

# --- Global variable for the QA Chain ---
# We will initialize this once on startup to avoid reloading the model on every request.
qa_chain = None


# --- Pydantic model for the request body ---
class PromptRequest(BaseModel):
    prompt: str


# --- Function to initialize the AI model and knowledge base ---
def initialize_ai_backend():
    global qa_chain
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found.")

        # 1. Load Knowledge Base
        all_documents = []
        try:
            txt_loader = DirectoryLoader('career_knowledge/', glob="**/*.txt", loader_cls=TextLoader,
                                         show_progress=True)
            txt_documents = txt_loader.load()
            all_documents.extend(txt_documents)
        except Exception:
            pass  # Ignore if no txt files
        try:
            csv_loader = DirectoryLoader('career_knowledge/', glob="**/*.csv", loader_cls=CSVLoader, show_progress=True)
            csv_documents = csv_loader.load()
            all_documents.extend(csv_documents)
        except Exception:
            pass  # Ignore if no csv files

        if not all_documents:
            raise ValueError("No documents found in 'career_knowledge' folder.")

        # 2. Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)

        # 3. Create embeddings and Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = Chroma.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 4. Set up the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.3)

        # 5. Create the RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False  # We only need the result for the API
        )
        print("✅ AI Backend initialized successfully!")

    except Exception as e:
        print(f"❌ Error during AI Backend initialization: {e}")
        qa_chain = None  # Ensure chain is None if setup fails


# --- FastAPI event handler ---
@app.on_event("startup")
async def startup_event():
    """This runs once when the server starts."""
    print("🚀 Server starting up...")
    initialize_ai_backend()


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "AI backend is running."}


@app.post("/get_recommendation")
async def get_recommendation(request: PromptRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="AI Backend is not initialized. Check server logs for errors.")

    try:
        user_prompt = request.prompt
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # Get the result from the AI
        result = qa_chain({"query": user_prompt})

        return {"recommendation": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
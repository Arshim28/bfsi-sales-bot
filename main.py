import nest_asyncio
nest_asyncio.apply()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bot.api import router as api_router  # Ensure this import path is correct
from bot.database import create_db_and_tables
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="BFSI Sales Bot Generator API", version="0.1.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Ensure data directories exist
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/parsed", exist_ok=True)
    os.makedirs("data/generations", exist_ok=True)
    os.makedirs("data/analyses", exist_ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router

api = FastAPI()

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api.include_router(
    router, 
    prefix="/api",
)
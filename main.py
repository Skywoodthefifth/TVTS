from typing import Annotated

import typer
import uvicorn
from dotenv import load_dotenv

from api import api

def start_api_server(port: Annotated[int, "Port to run the API server on"] = 8000):
    uvicorn.run(api, port=port)

if __name__ == "__main__":
    load_dotenv()
    typer.run(start_api_server)

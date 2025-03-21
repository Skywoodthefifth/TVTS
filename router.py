from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from google_model import GoogleModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

router = APIRouter()
google_model = GoogleModel()

@router.post("/v1/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    print(file)
    audio_file = await file.read()
    
    # Optional: Save the file to disk
    with open(file.filename, "wb") as f:
        f.write(audio_file)
        
    audio_file = open(file.filename, "rb")
    
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return JSONResponse(content={"transcription": transcription.text})

@router.post("/v1/llm")
async def llm_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    response = google_model.generate_content(message)
    return JSONResponse(content={"response": response})

@router.post("/v1/tts")
async def tts_endpoint(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"message": "Text input is required."}
            )
        
        speech_file_path = "output.wav"
        with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        ) as response:
            response.stream_to_file(speech_file_path)
        
        return FileResponse(
            path=speech_file_path,
            media_type="audio/wav",
            filename="output.wav"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error generating TTS: {str(e)}"}
        )

@router.post("/v1/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        print(file)
        # Read audio file contents (await async read)
        contents = await file.read()
        
        # Optional: Validate file type
        if not file.content_type.startswith("audio/"):
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid file type. Only audio files are allowed."}
            )
        
        # Optional: Save the file to disk
        with open(file.filename, "wb") as f:
            f.write(contents)
        
        # Return metadata
        return JSONResponse({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(contents),
            "message": "File uploaded successfully"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )

@router.post("/v1/clear-history")
async def clear_history_endpoint():
    google_model.clear_history()
    return JSONResponse(content={"message": "Conversation history cleared successfully."})
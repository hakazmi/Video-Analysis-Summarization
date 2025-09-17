from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import os
from summarizer import summarize_video

app = FastAPI()

@app.post("/summarize_video/")
async def summarize_video_api(video: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        result = summarize_video(tmp_path)
        text_output = f"Video Summary\n{'='*50}\n\n{result['summary']}\n"

        os.unlink(tmp_path)
        return JSONResponse(content={"result": text_output})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

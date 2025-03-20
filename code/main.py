from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os, shutil
import java.io.FileOutputStream
from my_api_project.model_inference import run_inference  # This function returns a dictionary with inference results

app = FastAPI()

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    # Save the uploaded file to a temporary folder
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, audio_file.filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    try:
        # Run inference on the saved file.
        # run_inference returns a dictionary like:
        # {"events_per_hour": ..., "osa_severity": ..., "total_events": ...}
        inference_result = run_inference(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)  # Clean up the temporary file
    
    return JSONResponse(content=inference_result)

# To run: uvicorn main:app --host 0.0.0.0 --port 8000

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AemroVision API",
    description="Brain Tumor MRI Classification Microservice",
    version="0.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


# Define the expected JSON payload schema using Pydantic
class ImageRequest(BaseModel):
    image: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/predict")
async def predict_route(payload: ImageRequest):
    try:
        # Decode the incoming base64 string into a physical file
        decodeImage(payload.image, clApp.filename)

        # Execute classification pipeline
        result = clApp.classifier.predict()

        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
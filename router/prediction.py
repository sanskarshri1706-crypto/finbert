from fastapi import APIRouter, HTTPException
from schema.request import SentimentRequest
from schema.response import SentimentResponse
from services.inference import GetInference

router = APIRouter(prefix="/sentiment",tags = ["sentiment_analysis"])


@router.get("/")
async def home():
    return {
        "message": "Welcome to Sentiment Analysis API"
    }


@router.post("/predict",response_model = SentimentResponse)
def predict_sentiment(request:SentimentRequest):
    model_pred = GetInference('model/model_finbert.pth')
    prediction = model_pred.predict(request.text)
    final_response = SentimentResponse(text = request.text,sentiment=prediction)
    return final_response
    
    
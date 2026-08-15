from pydantic import BaseModel


class SentimentResponse(BaseModel):
    text: str
    sentiment: str


    
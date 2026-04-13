from fastapi import FastAPI
from app.model_loader import get_model
from pydantic import BaseModel
from app.tokenizer import Tokenizer


app = FastAPI()


#Load assets once at startup
model, vocab = get_model()

#Initialize the tokenizer
tokenizer = Tokenizer(vocab_dict=vocab)


class PredictionRequest(BaseModel):
    text:str
    





@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict_sentiment(request: PredictionRequest):

    
    label, score = model.predict(request.text, tokenizer)
    
    return {
        "text": request.text,
        "sentiment": label,
        "probability": round(score, 4)
    }
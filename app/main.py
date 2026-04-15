from fastapi import FastAPI
from app.model_loader import get_model
from pydantic import BaseModel
from app.tokenizer import Tokenizer
from fastapi import HTTPException
from app.preprocessor import TextPreprocessor

app = FastAPI()


#Load assets once at startup
model, vocab = get_model()

#Initialize the tokenizer
tokenizer = Tokenizer(vocab_dict=vocab)
preprocessor = TextPreprocessor(tokenizer=tokenizer)


class PredictionRequest(BaseModel):
    text:str
    
    
class BatchPredictionRequest(BaseModel):
    texts:list[str]
    





@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict_sentiment(request: PredictionRequest):
    
    
    cleaned_text = preprocessor.clean(request.text)
    
    if not cleaned_text:
        return HTTPException(status_code=400, detail="Input cannot be empty.")

    
    label, score = model.predict(cleaned_text, tokenizer)
    
    return {
        "text": request.text,
        "sentiment": label,
        "probability": round(score, 4)
    }
    
    
# @app.post("/batch_predict")
# async def batch_predict_sentiment(request:BatchPredictionRequest):
    
#     cleaned_texts = []
    
#     for text in request.texts:
        
#         cleaned_text = preprocessor.clean(text)
        
#         if cleaned_text:
#             cleaned_texts.append(cleaned_text)
            
#     if not cleaned_texts:
#         return {"predictions": []}

#     # Unpack the three lists from the model
#     labels, scores, confidences = model.batch_predict(cleaned_texts, tokenizer)
    
    
#     results = [
#         {"sentiment": label, "confidence": conf, "positive_score": score}
#         for label, score, conf in zip(labels, scores, confidences)
#     ]
    
#     return {"predictions": results}


@app.post("/batch_predict")
async def batch_predict_sentiment(request: BatchPredictionRequest):
    # 1. identify valid strings and their original positions
    valid_tasks = []
    
    for i , text in enumerate(request.texts):
        cleaned = preprocessor.clean(text)
        
        if cleaned:
            valid_tasks.append((i,cleaned))
            
    #2. Handle the case where the whole batch is empty
    if not valid_tasks:
        return {
            "predictions":[
                {"error":"Empty input"} for _ in request.texts
            ]
        }
        
    #3. Separate the indices from the cleaned text
    indices, texts_to_predict = zip(*valid_tasks)
    
    #4. Run model only on valid texts
    labels, scores, confs = model.batch_predict(list(texts_to_predict),preprocessor)
    
    
    #5. Create a result list the same size as the original input
    final_result = [None] * len(request.texts)
    
    
    #6. fill in the predicted results using the saved indices.
    for i, label, score, conf in zip(indices, labels, scores, confs):
        final_result[i] = {
            "sentiment":label,
            "confidence": conf,
            "positive_score":score
        }
        
    #7. Fill any remaining None slots with an error placeholder
    for i in range(len(final_result)):
        if  final_result[i] is None:
            final_result[i] = {"sentiment": "N/A", "error": "Invalid or empty input"}

    return {"predictions": final_result}


        
@app.get("/examples")
async def get_examples():
    return {
        "example_request": {
            "texts": ["I love this!", "This is terrible."]
        },
        "example_response": {
            "predictions": [
                {"sentiment": "Positive", "confidence": 0.99},
                {"sentiment": "Negative", "confidence": 0.98}
            ]
        }
    }
            
            
            
   
# NLP FastAPI Sentiment Serving

Lightweight FastAPI service for binary sentiment inference using a PyTorch model and custom tokenizer.

## What it does
- Loads model config, weights, and vocabulary from `checkpoints/`
- Exposes REST endpoints for health/root and sentiment prediction
- Returns sentiment label (`Negative`/`Positive`) and positive-class probability

## Project layout
- `app/main.py`: FastAPI app and `/predict` endpoint
- `app/model_loader.py`: loads config, vocab, and model weights
- `app/model.py`: `SentimentModel` definition and `predict` helper
- `app/tokenizer.py`: text cleaning, tokenization, encode/decode
- `checkpoints/config.json`: model metadata and weights path
- `checkpoints/sentiment_model.pth`: trained model parameters
- `checkpoints/vocab.json`: vocabulary map used by tokenizer
- `requirements.txt`: Python dependencies

## Requirements
- Python 3.10+
- `pip`

## Local setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API usage
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I really enjoyed this movie"}'
```

Expected response keys: `text`, `sentiment`, `probability`.

## Notes
- `Dockerfile` exists but is currently empty.
- `tests/` is present but currently has no test files.

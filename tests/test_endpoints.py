import sys
from pathlib import Path

# Ensure project root is importable so `from app.main import app` works in pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"message": "Hello World"}

    
    
def test_batch_predict_logic():
    payload = {"texts": ["I love coding!", "This is a bug."]}
    response = client.post("/batch_predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check length
    assert len(data["predictions"]) == 2
    
    # 1. First prediction logic (Positive + High Confidence)
    assert data["predictions"][0]["sentiment"] == "Positive"
    assert data["predictions"][0]["confidence"] > 0.5  # Range-based 📈
    
    # 2. Second prediction logic (Negative + High Confidence)
    assert data["predictions"][1]["sentiment"] == "Negative"
    assert data["predictions"][1]["confidence"] > 0.5
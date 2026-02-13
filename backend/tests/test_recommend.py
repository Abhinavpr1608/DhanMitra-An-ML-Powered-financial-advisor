import json
from app import app

def test_health():
    client = app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

def test_recommend():
    client = app.test_client()
    payload = {
        "income": 60000,
        "expenses": 35000,
        "goal_amount": 200000,
        "duration_months": 24,
        "risk_level": "medium"
    }
    r = client.post("/api/recommend", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "plan" in data
    assert "product" in data["plan"]

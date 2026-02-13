[README.md](https://github.com/user-attachments/files/25294615/README.md)
# DhanMitra Backend (Flask + ML)

## Quickstart

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Train models (uses sample_data/dhanmitra_dataset.csv)
python -m model.train

# Run API
cp .env.example .env  # (optional) set keys
python app.py
# API runs on http://localhost:8000
```

## Endpoints

- `GET /api/health` – health check
- `POST /api/recommend` – input JSON:
```json
{
  "income": 60000,
  "expenses": 35000,
  "goal_amount": 200000,
  "duration_months": 24,
  "risk_level": "medium"
}
```
Returns recommended product, expected_return, and a simple explanation.

- `POST /api/chatbot` – input JSON: `{ "message": "Is FD better than SIP?" }`

## Tests
```bash
pytest -q
```

## Notes
- Replace `get_fd_rate_live()` with a real FD-rate source or other market data APIs.
- For real explainability, add SHAP to compute feature attributions per recommendation.

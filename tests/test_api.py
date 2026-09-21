from fastapi.testclient import TestClient
from app import app

# TestClient simulates web browser requests to your FastAPI app
client = TestClient(app)


def test_read_homepage():
    """Check that the web UI index page responds with status code 200 OK"""
    response = client.get("/")
    assert response.status_code == 200
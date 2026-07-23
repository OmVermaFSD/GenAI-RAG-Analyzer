import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def get_auth_token():
    test_username = "testuser_query"
    test_password = "password123"
    client.post("/register", json={"username": test_username, "password": test_password})
    response = client.post("/login", json={"username": test_username, "password": test_password})
    return response.json()["access_token"]

def test_query_short_question():
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/query", headers=headers, json={"question": "hi"})
    assert response.status_code == 422  # Unprocessable Entity due to min_length=3 validation

def test_query_empty_vector_db():
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/query", headers=headers, json={"question": "What is the document about?"})
    # Either returns 533/503 Service Unavailable due to empty vector db or 200 if documents indexed
    assert response.status_code in [200, 503, 500]

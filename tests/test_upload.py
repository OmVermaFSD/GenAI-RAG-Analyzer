import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def get_auth_token():
    test_username = "testuser_upload"
    test_password = "password123"
    client.post("/register", json={"username": test_username, "password": test_password})
    response = client.post("/login", json={"username": test_username, "password": test_password})
    return response.json()["access_token"]

def test_upload_invalid_file_type():
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("test.txt", b"This is text content", "text/plain")}

    response = client.post("/upload", headers=headers, files=files)
    assert response.status_code == 400
    assert "PDF files" in response.json()["detail"]

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_initialized" in data

def test_register_and_login():
    test_username = "testuser_auth"
    test_password = "password123"

    # Register user
    reg_response = client.post(
        "/register",
        json={"username": test_username, "password": test_password}
    )
    assert reg_response.status_code in [201, 400]

    # Login user
    login_response = client.post(
        "/login",
        json={"username": test_username, "password": test_password}
    )
    assert login_response.status_code == 200
    data = login_response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_unauthenticated_protected_endpoint():
    # Attempt upload without bearer token
    response = client.post("/upload")
    assert response.status_code in [401, 422]

@echo off
echo Starting DocuMiner...
docker-compose up --build -d
echo DocuMiner is running!
echo Java Backend: http://localhost:8080
echo Python OCR Engine: http://localhost:5000
pause

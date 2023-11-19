run_train:
	python train/train.py

test_server:
	uvicorn app.predict:app --reload --app-dir web_service --port 9696

test_client:
	curl -X POST -d @row_data.json -H "Content-Type: application/json" http://localhost:9696/predict

docker_build:
	docker build -t digit_recognizer:v1 -f web_service/Dockerfile .

docker_run:
	docker run -p 9696:9696 digit_recognizer:v1
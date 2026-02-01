from locust import HttpUser, task, between
import numpy as np
import json

class CreditScoringUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict_credit_score(self):
        # Генерация тестовых данных
        features = {
            "features": np.random.randn(23).tolist()
        }
        
        headers = {"Content-Type": "application/json"}
        self.client.post("/predict", json=features, headers=headers)
    
    @task(3)
    def health_check(self):
        self.client.get("/health")

# Запуск: locust -f locustfile.py --host=http://localhost:8000
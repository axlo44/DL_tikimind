import requests
import json

# Test de l'API
url = "http://localhost:8000/predict"

# Données de test : risque élevé : L'utilisateur n'a pas répondu correctement aux questions et n'a même pas validé la dernière question
test_data = {
    "session": {
        "user_id": "test_user_123",
        "actions": [
            {
                "action_type": "enter",
                "item_id": "b123",
                "timestamp": 1751362216 #01/07/2025 11h30
            },
            {
                "action_type": "respond",
                "item_id": "q123",
                "timestamp": 1751362876, #01/07/2025 11h40
                "user_answer": "A",
                "correct_answer": "B"
            },
            {
                "action_type": "submit",
                "item_id": "b123",
                "timestamp": 1751363416 #01/07/2025 11h50
            },
             {
                "action_type": "enter",
                "item_id": "b124",
                "timestamp": 1751364016 #01/07/2025 12h00
            },
            {
                "action_type": "respond",
                "item_id": "q124",
                "timestamp": 1751364616, #01/07/2025 12h10
                "user_answer": "A",
                "correct_answer": "B"
            }

        ]
    }
}

# Envoyer la requête
response = requests.post(url, json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pickle
import logging
from datetime import datetime
from .models.schemas import PredictionRequest, PredictionResponse, HealthResponse
from .services.prediction_service import PredictionService

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = FastAPI(
    title="API Prédiction d'Abandon des étudiants",
    description="API pour prédire le risque d'abandon des étudiants",
    version="1.0.0"
)

# CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
prediction_service = None

@app.on_event("startup")
async def load_model():
    """Charge le modèle au démarrage de l'application"""
    global model, prediction_service
    
    try:
        logger.info("Chargement du modèle...")
        
        # Charger le modèle TensorFlow
        model = tf.keras.models.load_model('api/models/abandon_prediction_model.keras')
        
        # Charger les encodeurs
        with open('api/models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Charger les métadonnées
        with open('api/models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Initialiser le service de prédiction
        prediction_service = PredictionService(model, encoders, metadata)
        
        logger.info("Modèle chargé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Point de santé de l'API"""
    return HealthResponse(
        status="OK",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de santé détaillée"""
    return HealthResponse(
        status="OK" if model is not None else "ERROR",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_abandon(request: PredictionRequest):
    """
    Prédit le risque d'abandon pour une session utilisateur
    """
    if prediction_service is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        session_data = {
            'user_id': request.session.user_id,
            'actions': [action.model_dump() for action in request.session.actions]
        }
        
        # Effectuer la prédiction
        result = prediction_service.predict_abandon(session_data)
           
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
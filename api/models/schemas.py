from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserAction(BaseModel):
    """Une action utilisateur"""
    action_type: str = Field(..., description="Type d'action (ex: 'answer', 'view')")
    item_id: str = Field(..., description="ID de l'item (question/bundle)")
    timestamp: int = Field(..., description="Timestamp en millisecondes")
    user_answer: Optional[str] = Field(None, description="Réponse de l'utilisateur")
    correct_answer: Optional[str] = Field(None, description="Bonne réponse")

class UserSession(BaseModel):
    """Session utilisateur complète"""
    user_id: str = Field(..., description="ID unique de l'utilisateur")
    actions: List[UserAction] = Field(..., description="Liste des actions", min_items=1)

class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    session: UserSession

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    user_id: str
    abandon_probability: float = Field(..., ge=0.0, le=1.0)
    abandon_prediction: bool
    confidence: str
    recommendation: str
    processed_actions: int

class HealthResponse(BaseModel):
    """Réponse de santé de l'API"""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime
import numpy as np
from typing import Dict, Any
from ..utils.preprocessing import preprocess_user_session

class PredictionService:
    def __init__(self, model, encoders: Dict, metadata: Dict):
        self.model = model
        self.encoders = encoders
        self.metadata = metadata
        self.threshold = metadata.get('threshold', 0.5)
    
    def predict_abandon(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédit le risque d'abandon pour une session utilisateur
        """
        try:
            # Préprocessing
            features = preprocess_user_session(session_data, self.encoders)
            
            if features is None:
                return {
                    'error': 'Données insuffisantes pour la prédiction',
                    'min_actions_required': 3
                }
            
            # Prédiction
            features_reshaped = features.reshape(1, *features.shape)
            probability = float(self.model.predict(features_reshaped, verbose=0)[0][0])
            
            # Déterminer la prédiction binaire
            prediction = probability > self.threshold
            
            # Niveau de confiance
            confidence = self._get_confidence_level(probability)
            
            # Recommandation
            recommendation = self._get_recommendation(probability, len(session_data['actions']))
            
            return {
                'user_id': session_data['user_id'],
                'abandon_probability': round(probability, 4),
                'abandon_prediction': prediction,
                'confidence': confidence,
                'recommendation': recommendation,
                'processed_actions': len(session_data['actions'])
            }
            
        except Exception as e:
            return {
                'error': f'Erreur lors de la prédiction: {str(e)}'
            }
    
    def _get_confidence_level(self, probability: float) -> str:
        """Détermine le niveau de confiance"""
        if probability > 0.8 or probability < 0.2:
            return "Élevée"
        elif probability > 0.65 or probability < 0.35:
            return "Moyenne"
        else:
            return "Faible"
    
    def _get_recommendation(self, probability: float, num_actions: int) -> str:
        """Génère une recommandation"""
        if probability > 0.7:
            return "🚨 Risque élevé d'abandon - Intervention recommandée"
        elif probability > 0.5:
            return "⚠️ Risque modéré - Surveillance conseillée"
        elif num_actions < 5:
            return "👀 Début de session - Continuer l'observation"
        else:
            return "✅ Utilisateur engagé - Pas d'intervention nécessaire"
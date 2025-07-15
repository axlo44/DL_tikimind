import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Dict, Any

def feature_comportemental(actions_df: pd.DataFrame, max_actions: int = 8) -> np.ndarray:
    """
    Version adaptée de votre fonction pour l'API
    """
    obs = actions_df.head(max_actions)
    
    if len(obs) < 3:
        return None
    
    # 1. Vitesse de réponse moyenne 
    delta_seq = obs['delta_t'].values
    delta_seq = delta_seq[~np.isnan(delta_seq)]
    response_speed = np.mean(delta_seq[delta_seq > 0]) if len(delta_seq) > 0 and np.any(delta_seq > 0) else 300
    
    # 2. Calcul de précision et tendance
    correct_seq = obs['correct'].values
    questions_mask = correct_seq >= 0
    
    if questions_mask.any():
        correct_answers = correct_seq[questions_mask]
        accuracy = np.mean(correct_answers) if len(correct_answers) > 0 else 0.5
        
        if len(correct_answers) > 2:
            try:
                x_vals = np.arange(len(correct_answers))
                corr_matrix = np.corrcoef(x_vals, correct_answers)
                perf_trend = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
            except:
                perf_trend = 0
        else:
            perf_trend = 0
    else:
        accuracy = 0.5
        perf_trend = 0
    
    # 3. Diversité des actions
    action_diversity = obs['action_id'].nunique() / len(obs) if len(obs) > 0 else 0
    
    # 4. Persistance
    if questions_mask.sum() > 1:
        failures = correct_seq[questions_mask] == 0
        persistence = 1.0 - np.mean(failures) if len(failures) > 0 else 0.5
    else:
        persistence = 0.5
    
    # Validation des valeurs
    accuracy = accuracy if np.isfinite(accuracy) else 0.5
    response_speed = response_speed if np.isfinite(response_speed) else 300
    perf_trend = perf_trend if np.isfinite(perf_trend) else 0
    action_diversity = action_diversity if np.isfinite(action_diversity) else 0
    persistence = persistence if np.isfinite(persistence) else 0.5
    
    # Créer la séquence
    sequence = []
    for i in range(len(obs)):
        delta_val = min(obs.iloc[i]['delta_t'], 600) if np.isfinite(obs.iloc[i]['delta_t']) else 0
        
        sequence.append([
            obs.iloc[i]['action_id'],
            obs.iloc[i]['type_id'],
            delta_val,
            obs.iloc[i]['correct'],
            accuracy,
            min(response_speed, 600),
            perf_trend,
            action_diversity,
            persistence,
            i / max_actions
        ])
    
    # Padding
    padded_sequence = pad_sequences(
        [sequence], 
        maxlen=max_actions,
        dtype='float32',
        padding='post',
        value=0.0
    )
    
    return padded_sequence[0]

def preprocess_user_session(session_data: Dict[str, Any], encoders: Dict) -> np.ndarray:
    """
    Convertit une session utilisateur en features pour le modèle
    """
    actions = session_data['actions']
    
    # Créer DataFrame
    df_actions = pd.DataFrame([
        {
            'action_type': action['action_type'],
            'item_id': action['item_id'],
            'timestamp': pd.to_datetime(action['timestamp'], unit='ms'),
            'user_answer': action.get('user_answer'),
            'correct_answer': action.get('correct_answer')
        }
        for action in actions
    ])
    
    # Trier par timestamp
    df_actions = df_actions.sort_values('timestamp')
    
    # Calculer delta_t
    df_actions['delta_t'] = df_actions['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Encoder action_type
    try:
        df_actions['action_id'] = encoders['action_encoder'].transform(df_actions['action_type'])
    except ValueError:
        # Action inconnue, utiliser une valeur par défaut
        df_actions['action_id'] = 0
    
    # Créer type à partir d'item_id
    df_actions['type'] = df_actions['item_id'].apply(
        lambda x: 'question' if str(x).startswith('q') else 'bundle'
    )
    
    # Encoder type
    try:
        df_actions['type_id'] = encoders['type_encoder'].transform(df_actions['type'])
    except ValueError:
        df_actions['type_id'] = 0
    
    # Calculer correct
    def create_correct_column(row):
        if row['type'] == 'question' and pd.notna(row['correct_answer']) and pd.notna(row['user_answer']):
            return 1 if str(row['user_answer']).strip() == str(row['correct_answer']).strip() else 0
        return -1
    
    df_actions['correct'] = df_actions.apply(create_correct_column, axis=1)
    
    # Générer les features comportementales
    features = feature_comportemental(df_actions, max_actions=8)
    
    return features
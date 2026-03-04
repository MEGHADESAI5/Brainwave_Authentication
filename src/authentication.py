"""
Authentication Engine Module
Handles user verification using trained models
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EEGAuthenticator:
    """
    Main authentication engine for EEG-based user verification
    """
    
    def __init__(self, models_dir='models/saved_models/'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.threshold = 0.7  # Confidence threshold for authentication
        self.user_templates = {}  # Store user EEG templates
        
    def load_models(self):
        """
        Load trained models from disk
        """
        try:
            # Load scaler if exists
            if os.path.exists(f'{self.models_dir}/scaler.pkl'):
                self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
            
            # Load ML models
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.pkl') and not file.startswith('scaler'):
                    model_name = file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(f'{self.models_dir}/{file}')
                    print(f"Loaded {model_name} model")
            
            # Load ANN model
            if os.path.exists(f'{self.models_dir}/ann_model.h5'):
                import tensorflow as tf
                self.models['ann'] = tf.keras.models.load_model(f'{self.models_dir}/ann_model.h5')
                print("Loaded ANN model")
            
            print(f"Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def register_user(self, user_id, eeg_features):
        """
        Register a new user by storing their EEG template
        
        Args:
            user_id: Unique user identifier
            eeg_features: Extracted EEG features for the user
        """
        # Normalize features
        features_normalized = self.scaler.fit_transform(eeg_features.reshape(1, -1))
        
        # Store user template
        self.user_templates[user_id] = {
            'features': features_normalized.flatten(),
            'samples': [features_normalized.flatten()]
        }
        
        print(f"User {user_id} registered successfully")
        return True
    
    def authenticate(self, user_id, eeg_features, model_name='random_forest'):
        """
        Authenticate a user based on EEG features
        
        Args:
            user_id: Claimed user identity
            eeg_features: Extracted EEG features from current session
            model_name: Which model to use for authentication
            
        Returns:
            authentication_result: True/False
            confidence: Confidence score
            details: Additional information
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return False, 0.0, {"error": "Model not found"}
        
        # Normalize features
        features_normalized = self.scaler.transform(eeg_features.reshape(1, -1))
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        if model_name == 'ann':
            # ANN gives probability distribution
            probabilities = model.predict(features_normalized)[0]
            predicted_user = np.argmax(probabilities)
            confidence = probabilities[predicted_user]
        else:
            # ML models
            predicted_user = model.predict(features_normalized)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_normalized)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 1.0 if predicted_user == user_id else 0.0
        
        # Determine if authentication successful
        is_authenticated = (predicted_user == user_id) and (confidence >= self.threshold)
        
        details = {
            'claimed_user': user_id,
            'predicted_user': predicted_user,
            'confidence': confidence,
            'threshold': self.threshold,
            'model_used': model_name
        }
        
        return is_authenticated, confidence, details
    
    def continuous_authentication(self, user_id, eeg_stream, window_size=10, model_name='random_forest'):
        """
        Continuous authentication over a stream of EEG data
        
        Args:
            user_id: Claimed user identity
            eeg_stream: List of EEG feature vectors over time
            window_size: Number of samples to consider
            model_name: Which model to use
        """
        authentication_scores = []
        
        for i, features in enumerate(eeg_stream):
            is_auth, conf, _ = self.authenticate(user_id, features, model_name)
            authentication_scores.append({
                'timestamp': i,
                'authenticated': is_auth,
                'confidence': conf
            })
        
        # Calculate overall authentication status
        if len(authentication_scores) >= window_size:
            recent_scores = authentication_scores[-window_size:]
            auth_rate = sum([s['authenticated'] for s in recent_scores]) / window_size
            
            # User is considered authenticated if >70% of recent windows are authenticated
            is_continuously_auth = auth_rate >= 0.7
            
            return is_continuously_auth, authentication_scores
        else:
            return False, authentication_scores
    
    def calculate_far_frr(self, genuine_users, impostor_users, model_name='random_forest'):
        """
        Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        
        Args:
            genuine_users: List of (user_id, features) for genuine attempts
            impostor_users: List of (claim_id, actual_id, features) for impostor attempts
        """
        false_acceptances = 0
        false_rejections = 0
        genuine_attempts = len(genuine_users)
        impostor_attempts = len(impostor_users)
        
        # Test genuine users (should be accepted)
        for user_id, features in genuine_users:
            is_auth, _, _ = self.authenticate(user_id, features, model_name)
            if not is_auth:
                false_rejections += 1
        
        # Test impostor users (should be rejected)
        for claim_id, actual_id, features in impostor_users:
            is_auth, _, _ = self.authenticate(claim_id, features, model_name)
            if is_auth:
                false_acceptances += 1
        
        # Calculate rates
        far = false_acceptances / impostor_attempts if impostor_attempts > 0 else 0
        frr = false_rejections / genuine_attempts if genuine_attempts > 0 else 0
        
        # Calculate Equal Error Rate (EER) approximation
        eer = (far + frr) / 2
        
        metrics = {
            'FAR': far,
            'FRR': frr,
            'EER': eer,
            'false_acceptances': false_acceptances,
            'false_rejections': false_rejections,
            'total_genuine': genuine_attempts,
            'total_impostor': impostor_attempts
        }
        
        return metrics
    
    def save_authenticator_state(self, save_dir='models/authenticator/'):
        """
        Save authenticator state (templates and scaler)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save user templates
        template_data = {
            'user_templates': self.user_templates,
            'threshold': self.threshold
        }
        joblib.dump(template_data, f'{save_dir}/user_templates.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{save_dir}/scaler.pkl')
        
        print(f"Authenticator state saved to {save_dir}")
    
    def load_authenticator_state(self, load_dir='models/authenticator/'):
        """
        Load authenticator state
        """
        try:
            # Load templates
            if os.path.exists(f'{load_dir}/user_templates.pkl'):
                template_data = joblib.load(f'{load_dir}/user_templates.pkl')
                self.user_templates = template_data['user_templates']
                self.threshold = template_data['threshold']
            
            # Load scaler
            if os.path.exists(f'{load_dir}/scaler.pkl'):
                self.scaler = joblib.load(f'{load_dir}/scaler.pkl')
            
            print(f"Authenticator state loaded from {load_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading authenticator state: {e}")
            return False
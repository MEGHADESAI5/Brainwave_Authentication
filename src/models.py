"""
Machine Learning Models for EEG Authentication
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib
import os

class EEGAuthenticationModels:
    """
    Collection of machine learning models for EEG-based authentication
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize all ML models
        """
        # SVM with RBF kernel
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        # K-Nearest Neighbors
        self.models['knn'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        
        # Decision Tree (baseline)
        self.models['decision_tree'] = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            random_state=self.random_state
        )
        
        print(f"Initialized {len(self.models)} traditional ML models")
        return self.models
    
    def train_ml_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all ML models
        """
        if not self.models:
            self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Store trained model
            self.trained_models[name] = model
            
            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            print(f"{name} - Training Accuracy: {train_acc:.4f}")
            
            # Evaluate on validation set if provided
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                print(f"{name} - Validation Accuracy: {val_acc:.4f}")
                
                # Store results
                self.results[name] = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'model': model
                }
    
    def create_ann_model(self, input_dim, num_classes):
        """
        Create Artificial Neural Network for EEG authentication
        
        Args:
            input_dim: Number of input features
            num_classes: Number of users to authenticate
        """
        model = models.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layer 1
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layer 2
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Hidden layer 3
            layers.Dense(16, activation='relu'),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ann(self, X_train, y_train, X_val, y_val, 
                  epochs=50, batch_size=32, save_best=True):
        """
        Train Artificial Neural Network
        """
        print("\n" + "="*50)
        print("Training Artificial Neural Network")
        print("="*50)
        
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Create model
        ann_model = self.create_ann_model(input_dim, num_classes)
        
        # Print model summary
        ann_model.summary()
        
        # Callbacks
        callbacks = []
        
        if save_best:
            checkpoint = keras.callbacks.ModelCheckpoint(
                'models/saved_models/best_ann_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Train model
        history = ann_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.trained_models['ann'] = ann_model
        self.results['ann'] = {
            'history': history.history,
            'model': ann_model
        }
        
        return ann_model, history
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            print(f"\nEvaluating {name.upper()}...")
            
            if name == 'ann':
                # ANN evaluation
                y_pred_prob = model.predict(X_test)
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                # ML model evaluation
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    def save_models(self, save_dir='models/saved_models/'):
        """
        Save trained models to disk
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            if name == 'ann':
                # Save ANN model
                model.save(f'{save_dir}/ann_model.h5')
                print(f"ANN model saved to {save_dir}/ann_model.h5")
            else:
                # Save ML models
                joblib.dump(model, f'{save_dir}/{name}_model.pkl')
                print(f"{name} model saved to {save_dir}/{name}_model.pkl")
    
    def load_models(self, load_dir='models/saved_models/'):
        """
        Load trained models from disk
        """
        self.trained_models = {}
        
        # Load ML models
        ml_model_files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
        for model_file in ml_model_files:
            name = model_file.replace('_model.pkl', '')
            self.trained_models[name] = joblib.load(f'{load_dir}/{model_file}')
            print(f"Loaded {name} model")
        
        # Load ANN model
        if os.path.exists(f'{load_dir}/ann_model.h5'):
            self.trained_models['ann'] = keras.models.load_model(f'{load_dir}/ann_model.h5')
            print("Loaded ANN model")
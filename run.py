"""
Main script to run the complete Brainwave Authentication System
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.data_loader import EEGDataLoader
from src.preprocessing import EEGPreprocessor
from src.feature_extraction import EEGFeatureExtractor
from src.models import EEGAuthenticationModels
from src.authentication import EEGAuthenticator
from src.visualization import EEGVisualizer

def main():
    """
    Main function to execute the complete EEG authentication pipeline
    """
    print("="*60)
    print("BRAINWAVE AUTHENTICATION SYSTEM")
    print("EEG-Based Biometric Verification")
    print("="*60)
    
    # Step 1: Initialize components
    print("\n[1] Initializing system components...")
    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=256)
    feature_extractor = EEGFeatureExtractor(sampling_rate=256)
    visualizer = EEGVisualizer()
    
    # Step 2: Load/Generate dataset
    print("\n[2] Loading EEG dataset...")
    X, y = data_loader.load_sample_dataset(
        n_users=8,           # 8 different users
        samples_per_user=40, # 40 samples per user
        n_channels=14        # 14 EEG channels
    )
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
        test_size=0.2, val_size=0.1
    )
    
    # Step 3: Visualize raw EEG signals
    print("\n[3] Visualizing raw EEG signals...")
    sample_idx = 0
    visualizer.plot_raw_eeg(
        X[sample_idx], 
        title=f"Raw EEG Signals - User {y[sample_idx]}"
    )
    
    # Step 4: Preprocess EEG signals
    print("\n[4] Preprocessing EEG signals...")
    
    # Preprocess a sample to show the effect
    sample_eeg = X[0]
    preprocessed_sample = preprocessor.preprocess_pipeline(
        sample_eeg,
        apply_notch=True,
        apply_bandpass=True,
        apply_artifact=True,
        apply_wavelet=True
    )
    
    # Visualize preprocessing results
    visualizer.plot_preprocessed_vs_raw(sample_eeg, preprocessed_sample, channel_idx=0)
    
    # Preprocess all training data
    print("\nPreprocessing all training data...")
    X_train_processed = []
    for eeg_sample in X_train:
        processed = preprocessor.preprocess_pipeline(
            eeg_sample,
            apply_notch=True,
            apply_bandpass=True,
            apply_artifact=True,
            apply_wavelet=True
        )
        X_train_processed.append(processed)
    
    X_train_processed = np.array(X_train_processed)
    
    # Preprocess validation and test data
    X_val_processed = []
    for eeg_sample in X_val:
        processed = preprocessor.preprocess_pipeline(
            eeg_sample,
            apply_notch=True,
            apply_bandpass=True,
            apply_artifact=True,
            apply_wavelet=True
        )
        X_val_processed.append(processed)
    X_val_processed = np.array(X_val_processed)
    
    X_test_processed = []
    for eeg_sample in X_test:
        processed = preprocessor.preprocess_pipeline(
            eeg_sample,
            apply_notch=True,
            apply_bandpass=True,
            apply_artifact=True,
            apply_wavelet=True
        )
        X_test_processed.append(processed)
    X_test_processed = np.array(X_test_processed)
    
    # Step 5: Feature Extraction
    print("\n[5] Extracting features from EEG signals...")
    
    # Extract features for all samples
    X_train_features = []
    for eeg_sample in X_train_processed:
        features = feature_extractor.extract_all_features(eeg_sample)
        X_train_features.append(features)
    X_train_features = np.array(X_train_features)
    
    X_val_features = []
    for eeg_sample in X_val_processed:
        features = feature_extractor.extract_all_features(eeg_sample)
        X_val_features.append(features)
    X_val_features = np.array(X_val_features)
    
    X_test_features = []
    for eeg_sample in X_test_processed:
        features = feature_extractor.extract_all_features(eeg_sample)
        X_test_features.append(features)
    X_test_features = np.array(X_test_features)
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Visualize frequency spectrum
    visualizer.plot_frequency_spectrum(X[0], sampling_rate=256, channel_idx=0)
    
    # Step 6: Train Machine Learning Models
    print("\n[6] Training machine learning models...")
    
    # Initialize models
    model_trainer = EEGAuthenticationModels(random_state=42)
    model_trainer.initialize_models()
    
    # Train ML models
    model_trainer.train_ml_models(X_train_features, y_train, X_val_features, y_val)
    
    # Train ANN
    ann_model, history = model_trainer.train_ann(
        X_train_features, y_train,
        X_val_features, y_val,
        epochs=30,
        batch_size=16
    )
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Step 7: Evaluate Models
    print("\n[7] Evaluating models on test set...")
    evaluation_results = model_trainer.evaluate_models(X_test_features, y_test)
    
    # Plot model comparison
    visualizer.plot_model_comparison(evaluation_results)
    
    # Plot confusion matrix for best model (Random Forest)
    if 'random_forest' in evaluation_results:
        y_pred = evaluation_results['random_forest']['predictions']
        visualizer.plot_confusion_matrix(
            y_test, y_pred,
            title="Confusion Matrix - Random Forest Classifier"
        )
    
    # Save models
    print("\n[8] Saving trained models...")
    model_trainer.save_models()
    
    # Step 8: Authentication Simulation
    print("\n[9] Simulating authentication process...")
    
    # Initialize authenticator
    authenticator = EEGAuthenticator()
    authenticator.models = model_trainer.trained_models
    
    # Register a user (using first user's data)
    user_id = 0
    user_template = X_train_features[y_train == user_id][0]
    authenticator.register_user(user_id, user_template)
    
    # Test genuine authentication
    print("\n--- Testing Genuine User ---")
    genuine_sample = X_test_features[y_test == user_id][0]
    is_auth, confidence, details = authenticator.authenticate(
        user_id, genuine_sample, model_name='random_forest'
    )
    print(f"Authentication Result: {'✓ AUTHORIZED' if is_auth else '✗ UNAUTHORIZED'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Details: {details}")
    
    # Visualize result
    visualizer.plot_authentication_result(
        is_auth, confidence, 
        authenticator.threshold, 
        f"User {user_id}"
    )
    
    # Test impostor authentication
    print("\n--- Testing Impostor User ---")
    # Find a different user
    impostor_id = 1 if len(np.unique(y_test)) > 1 else 0
    impostor_sample = X_test_features[y_test == impostor_id][0]
    
    is_auth, confidence, details = authenticator.authenticate(
        user_id, impostor_sample, model_name='random_forest'
    )
    print(f"Authentication Result: {'✓ AUTHORIZED' if is_auth else '✗ UNAUTHORIZED'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Details: {details}")
    
    # Visualize result
    visualizer.plot_authentication_result(
        is_auth, confidence, 
        authenticator.threshold, 
        f"User {user_id} (Impostor Attempt)"
    )
    
    # Step 9: Calculate FAR and FRR
    print("\n[10] Calculating FAR and FRR metrics...")
    
    # Prepare genuine attempts
    genuine_attempts = []
    for user in np.unique(y_test):
        user_samples = X_test_features[y_test == user]
        for sample in user_samples[:5]:  # Use first 5 samples per user
            genuine_attempts.append((user, sample))
    
    # Prepare impostor attempts
    impostor_attempts = []
    for i, user_i in enumerate(np.unique(y_test)):
        for j, user_j in enumerate(np.unique(y_test)):
            if i != j:  # Different users
                user_j_samples = X_test_features[y_test == user_j]
                for sample in user_j_samples[:3]:  # Use 3 samples
                    impostor_attempts.append((user_i, user_j, sample))
    
    # Calculate metrics
    metrics = authenticator.calculate_far_frr(
        genuine_attempts, impostor_attempts, model_name='random_forest'
    )
    
    print("\n" + "="*50)
    print("AUTHENTICATION METRICS")
    print("="*50)
    print(f"False Acceptance Rate (FAR): {metrics['FAR']:.4f}")
    print(f"False Rejection Rate (FRR): {metrics['FRR']:.4f}")
    print(f"Equal Error Rate (EER): {metrics['EER']:.4f}")
    print(f"Total Genuine Attempts: {metrics['total_genuine']}")
    print(f"Total Impostor Attempts: {metrics['total_impostor']}")
    print(f"False Acceptances: {metrics['false_acceptances']}")
    print(f"False Rejections: {metrics['false_rejections']}")
    
    # Save authenticator state
    authenticator.save_authenticator_state()
    
    print("\n" + "="*60)
    print("BRAINWAVE AUTHENTICATION SYSTEM - COMPLETED")
    print("="*60)
    
    return {
        'models': model_trainer.trained_models,
        'evaluation': evaluation_results,
        'metrics': metrics,
        'authenticator': authenticator
    }

if __name__ == "__main__":
    results = main()
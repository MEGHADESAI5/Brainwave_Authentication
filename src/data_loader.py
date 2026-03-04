"""
EEG Data Loader Module
Handles loading and initial processing of EEG datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EEGDataLoader:
    """
    Class to load and manage EEG dataset for authentication system
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.labels = None
        self.user_ids = None
        
    def load_sample_dataset(self, n_users=10, samples_per_user=50, n_channels=14):
        """
        Generate synthetic EEG dataset for simulation
        This mimics real EEG data patterns
        
        Args:
            n_users: Number of unique users
            samples_per_user: Number of EEG samples per user
            n_channels: Number of EEG channels
            
        Returns:
            X: EEG data array
            y: User labels
        """
        np.random.seed(42)
        
        X = []
        y = []
        
        for user_id in range(n_users):
            # Generate unique brainwave pattern for each user
            user_seed = user_id * 100
            
            for sample in range(samples_per_user):
                # Create EEG signal with user-specific characteristics
                eeg_sample = self._generate_eeg_sample(
                    n_channels=n_channels,
                    user_id=user_id,
                    sample_variation=sample
                )
                X.append(eeg_sample)
                y.append(user_id)
        
        self.data = np.array(X)
        self.labels = np.array(y)
        self.user_ids = np.unique(y)
        
        print(f"Dataset created: {self.data.shape[0]} samples, {self.data.shape[1]} channels")
        print(f"Number of users: {len(self.user_ids)}")
        
        return self.data, self.labels
    
    def _generate_eeg_sample(self, n_channels=14, user_id=0, sample_variation=0):
        """
        Generate a single EEG sample with user-specific patterns
        
        Brainwave frequency bands:
        - Delta (0.5-4 Hz): Deep sleep
        - Theta (4-8 Hz): Drowsiness, meditation
        - Alpha (8-12 Hz): Relaxed, eyes closed
        - Beta (12-30 Hz): Active thinking, focus
        - Gamma (30-50 Hz): Peak performance
        """
        # Time points (1 second of data at 256 Hz)
        time = np.linspace(0, 1, 256)
        
        eeg_sample = []
        
        for channel in range(n_channels):
            # Base signal with user-specific characteristics
            user_fingerprint = user_id * 0.1 + channel * 0.05
            
            # Generate different frequency components
            delta = 0.5 * np.sin(2 * np.pi * 2 * time + user_fingerprint)
            theta = 0.3 * np.sin(2 * np.pi * 6 * time + 2 * user_fingerprint)
            alpha = 0.8 * np.sin(2 * np.pi * 10 * time + 3 * user_fingerprint)
            beta = 0.4 * np.sin(2 * np.pi * 20 * time + 4 * user_fingerprint)
            gamma = 0.2 * np.sin(2 * np.pi * 40 * time + 5 * user_fingerprint)
            
            # Combine frequency bands (user-specific mixing)
            signal = (delta * 0.2 + 
                     theta * 0.3 + 
                     alpha * 0.8 + 
                     beta * 0.5 + 
                     gamma * 0.3)
            
            # Add some noise
            noise = np.random.normal(0, 0.1, len(time))
            signal += noise
            
            # Add sample variation (different mental states)
            variation = sample_variation * 0.01
            signal += variation * np.sin(2 * np.pi * 15 * time)
            
            eeg_sample.append(signal)
        
        return np.array(eeg_sample)
    
    def load_real_dataset(self, file_path):
        """
        Load real EEG dataset from CSV or other format
        """
        try:
            # Example: Load from CSV
            df = pd.read_csv(file_path)
            self.data = df.iloc[:, :-1].values
            self.labels = df.iloc[:, -1].values
            self.user_ids = np.unique(self.labels)
            
            print(f"Loaded real dataset: {self.data.shape}")
            return self.data, self.labels
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_sample_dataset() first.")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )
        
        # Second split: train and val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
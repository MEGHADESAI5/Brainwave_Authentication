"""
EEG Signal Preprocessing Module
Handles noise removal, filtering, and artifact elimination
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import pywt

class EEGPreprocessor:
    """
    Class to preprocess EEG signals for authentication
    """
    
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.filtered_data = None
        
    def bandpass_filter(self, data, lowcut=0.5, highcut=50, order=4):
        """
        Apply bandpass filter to remove noise outside brainwave frequencies
        
        Args:
            data: EEG signal array
            lowcut: Lower frequency bound (Hz)
            highcut: Upper frequency bound (Hz)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):  # For each channel
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def notch_filter(self, data, notch_freq=50, quality_factor=30):
        """
        Apply notch filter to remove power line interference (50/60 Hz)
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = notch_freq / nyquist
        
        b, a = iirnotch(w0, quality_factor)
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def remove_artifacts(self, data, threshold=3):
        """
        Remove artifacts using z-score thresholding
        Detects and removes eye blinks and muscle movements
        """
        cleaned = data.copy()
        
        for i in range(data.shape[0]):  # For each channel
            # Calculate z-scores
            channel_mean = np.mean(data[i])
            channel_std = np.std(data[i])
            z_scores = np.abs((data[i] - channel_mean) / channel_std)
            
            # Find artifact indices
            artifact_indices = np.where(z_scores > threshold)[0]
            
            if len(artifact_indices) > 0:
                # Interpolate artifacts
                non_artifact = np.where(z_scores <= threshold)[0]
                if len(non_artifact) > 1:
                    cleaned[i, artifact_indices] = np.interp(
                        artifact_indices,
                        non_artifact,
                        data[i, non_artifact]
                    )
        
        return cleaned
    
    def wavelet_denoising(self, data, wavelet='db4', level=4):
        """
        Apply wavelet denoising to remove noise while preserving signal features
        """
        denoised = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # Decompose signal
            coeffs = pywt.wavedec(data[i], wavelet, level=level)
            
            # Calculate threshold (universal threshold)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data[i])))
            
            # Apply soft thresholding to detail coefficients
            coeffs_thresh = list(coeffs)
            for j in range(1, len(coeffs_thresh)):
                coeffs_thresh[j] = pywt.threshold(coeffs_thresh[j], threshold, mode='soft')
            
            # Reconstruct signal
            denoised[i] = pywt.waverec(coeffs_thresh, wavelet)
            
            # Trim to original length if necessary
            if len(denoised[i]) > len(data[i]):
                denoised[i] = denoised[i][:len(data[i])]
        
        return denoised
    
    def normalize_signal(self, data, method='zscore'):
        """
        Normalize EEG signals
        
        Args:
            data: EEG signal
            method: 'zscore', 'minmax', or 'robust'
        """
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            if method == 'zscore':
                # Z-score normalization
                normalized[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
                
            elif method == 'minmax':
                # Min-max normalization
                normalized[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
                
            elif method == 'robust':
                # Robust normalization (using median and IQR)
                median = np.median(data[i])
                q75, q25 = np.percentile(data[i], [75, 25])
                iqr = q75 - q25
                normalized[i] = (data[i] - median) / iqr
        
        return normalized
    
    def preprocess_pipeline(self, data, apply_notch=True, apply_bandpass=True, 
                           apply_artifact=True, apply_wavelet=True):
        """
        Complete preprocessing pipeline
        """
        processed_data = data.copy()
        
        # Step 1: Bandpass filter
        if apply_bandpass:
            print("Applying bandpass filter...")
            processed_data = self.bandpass_filter(processed_data)
        
        # Step 2: Notch filter
        if apply_notch:
            print("Applying notch filter...")
            processed_data = self.notch_filter(processed_data)
        
        # Step 3: Artifact removal
        if apply_artifact:
            print("Removing artifacts...")
            processed_data = self.remove_artifacts(processed_data)
        
        # Step 4: Wavelet denoising
        if apply_wavelet:
            print("Applying wavelet denoising...")
            processed_data = self.wavelet_denoising(processed_data)
        
        # Step 5: Normalization
        print("Normalizing signals...")
        processed_data = self.normalize_signal(processed_data, method='zscore')
        
        self.filtered_data = processed_data
        print("Preprocessing complete!")
        
        return processed_data
"""
Feature Extraction Module
Extracts meaningful features from EEG signals for authentication
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from scipy.stats import entropy

class EEGFeatureExtractor:
    """
    Extract features from EEG signals for biometric authentication
    """
    
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.feature_names = []
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 50)
        }
    
    def extract_bandpower(self, data):
        """
        Extract power in different frequency bands
        """
        bandpowers = []
        
        for channel in data:
            # Compute power spectral density
            freqs, psd = signal.welch(channel, fs=self.sampling_rate, nperseg=256)
            
            channel_bands = []
            for band, (low, high) in self.bands.items():
                # Find indices corresponding to frequency band
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.trapz(psd[idx], freqs[idx])
                channel_bands.append(band_power)
            
            bandpowers.append(channel_bands)
        
        return np.array(bandpowers)
    
    def extract_statistical_features(self, data):
        """
        Extract statistical features from EEG signals
        """
        stats_features = []
        
        for channel in data:
            channel_stats = [
                np.mean(channel),           # Mean
                np.std(channel),             # Standard deviation
                np.var(channel),              # Variance
                stats.skew(channel),          # Skewness
                stats.kurtosis(channel),      # Kurtosis
                np.median(channel),           # Median
                np.max(channel) - np.min(channel),  # Range
                np.percentile(channel, 25),   # Q1
                np.percentile(channel, 75),   # Q3
                stats.iqr(channel)            # Interquartile range
            ]
            stats_features.append(channel_stats)
        
        return np.array(stats_features)
    
    def extract_frequency_features(self, data):
        """
        Extract frequency domain features using FFT
        """
        freq_features = []
        
        for channel in data:
            # Compute FFT
            channel_fft = fft(channel)
            freqs = fftfreq(len(channel), 1/self.sampling_rate)
            
            # Take only positive frequencies
            pos_idx = freqs > 0
            channel_fft = channel_fft[pos_idx]
            freqs = freqs[pos_idx]
            
            # Extract frequency features
            magnitude = np.abs(channel_fft)
            
            # Spectral features
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
            
            # Peak frequencies
            peak_freq = freqs[np.argmax(magnitude)]
            
            # Spectral energy
            energy = np.sum(magnitude ** 2)
            
            freq_features.append([
                spectral_centroid,
                spectral_spread,
                peak_freq,
                energy,
                np.mean(magnitude),
                np.std(magnitude)
            ])
        
        return np.array(freq_features)
    
    def extract_entropy_features(self, data):
        """
        Extract entropy-based features (signal complexity)
        """
        entropy_features = []
        
        for channel in data:
            # Approximate entropy (simplified)
            # Sample entropy
            def sample_entropy(time_series, m=2, r=0.2):
                N = len(time_series)
                def _maxdist(xi, xj):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = [time_series[i:i+m] for i in range(N - m + 1)]
                    B = 0
                    for i, pattern_i in enumerate(patterns):
                        for j, pattern_j in enumerate(patterns):
                            if i != j and _maxdist(pattern_i, pattern_j) <= r:
                                B += 1
                    return B / (N - m)
                
                return -np.log(_phi(m+1) / _phi(m))
            
            try:
                samp_ent = sample_entropy(channel)
            except:
                samp_ent = 0
            
            # Shannon entropy of power spectrum
            freqs, psd = signal.welch(channel, fs=self.sampling_rate)
            psd_norm = psd / np.sum(psd)
            shannon_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            entropy_features.append([samp_ent, shannon_ent])
        
        return np.array(entropy_features)
    
    def extract_wavelet_features(self, data, wavelet='db4', level=4):
        """
        Extract wavelet coefficients as features
        """
        wavelet_features = []
        
        for channel in data:
            # Decompose signal
            coeffs = pywt.wavedec(channel, wavelet, level=level)
            
            # Calculate energy of each decomposition level
            energies = []
            for coeff in coeffs:
                energy = np.sum(np.array(coeff) ** 2) / len(coeff)
                energies.append(energy)
            
            wavelet_features.append(energies)
        
        return np.array(wavelet_features)
    
    def extract_hjorth_parameters(self, data):
        """
        Extract Hjorth parameters (activity, mobility, complexity)
        """
        hjorth_features = []
        
        for channel in data:
            # First derivative
            dx = np.diff(channel)
            
            # Second derivative
            ddx = np.diff(dx)
            
            # Activity (variance)
            activity = np.var(channel)
            
            # Mobility (sqrt of variance of first derivative / variance of signal)
            mobility = np.sqrt(np.var(dx) / np.var(channel))
            
            # Complexity (mobility of first derivative / mobility of signal)
            complexity = np.sqrt(np.var(ddx) / np.var(dx)) / mobility if mobility != 0 else 0
            
            hjorth_features.append([activity, mobility, complexity])
        
        return np.array(hjorth_features)
    
    def extract_all_features(self, data):
        """
        Extract all features and combine them into a single feature vector
        """
        all_features = []
        
        print("Extracting bandpower features...")
        bandpower = self.extract_bandpower(data)
        
        print("Extracting statistical features...")
        statistical = self.extract_statistical_features(data)
        
        print("Extracting frequency features...")
        frequency = self.extract_frequency_features(data)
        
        print("Extracting entropy features...")
        entropy_feat = self.extract_entropy_features(data)
        
        print("Extracting wavelet features...")
        wavelet = self.extract_wavelet_features(data)
        
        print("Extracting Hjorth parameters...")
        hjorth = self.extract_hjorth_parameters(data)
        
        # Combine features for each channel
        for i in range(len(data)):
            channel_features = np.concatenate([
                bandpower[i].flatten(),
                statistical[i].flatten(),
                frequency[i].flatten(),
                entropy_feat[i].flatten(),
                wavelet[i].flatten(),
                hjorth[i].flatten()
            ])
            all_features.append(channel_features)
        
        # Average across channels for final feature vector
        final_features = np.mean(all_features, axis=0)
        
        # Create feature names for reference
        self.feature_names = self._generate_feature_names()
        
        print(f"Total features extracted: {len(final_features)}")
        
        return final_features
    
    def _generate_feature_names(self):
        """
        Generate names for all features (for documentation)
        """
        names = []
        
        # Bandpower names
        for band in self.bands.keys():
            names.append(f'bandpower_{band}')
        
        # Statistical names
        stat_names = ['mean', 'std', 'var', 'skew', 'kurtosis', 'median', 'range', 'q1', 'q3', 'iqr']
        names.extend(stat_names)
        
        # Frequency names
        freq_names = ['spectral_centroid', 'spectral_spread', 'peak_freq', 'energy', 'freq_mean', 'freq_std']
        names.extend(freq_names)
        
        # Entropy names
        entropy_names = ['sample_entropy', 'shannon_entropy']
        names.extend(entropy_names)
        
        # Wavelet names (assuming 5 levels)
        wavelet_names = [f'wavelet_energy_L{i}' for i in range(5)]
        names.extend(wavelet_names)
        
        # Hjorth names
        hjorth_names = ['activity', 'mobility', 'complexity']
        names.extend(hjorth_names)
        
        return names
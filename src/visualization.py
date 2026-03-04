"""
Visualization Module for EEG Authentication System
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
from sklearn.metrics import confusion_matrix, roc_curve, auc

class EEGVisualizer:
    """
    Class to handle all visualizations for the EEG authentication system
    """
    
    def __init__(self, style='default'):
        plt.style.use(style)
        self.figsize = (12, 8)
        
    def plot_raw_eeg(self, eeg_data, channel_names=None, title="Raw EEG Signals"):
        """
        Plot raw EEG signals for multiple channels
        """
        n_channels = eeg_data.shape[0]
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if n_channels == 1:
            axes = [axes]
        
        for i in range(n_channels):
            axes[i].plot(eeg_data[i], color='blue', linewidth=0.8)
            axes[i].set_ylabel(f'Ch {i+1}' if not channel_names else channel_names[i])
            axes[i].grid(True, alpha=0.3)
            
            # Add some styling
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        axes[-1].set_xlabel('Time (samples)')
        plt.tight_layout()
        plt.show()
        
    def plot_preprocessed_vs_raw(self, raw_data, preprocessed_data, channel_idx=0):
        """
        Compare raw and preprocessed EEG signals
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Raw signal
        ax1.plot(raw_data[channel_idx], color='red', alpha=0.7, linewidth=0.8)
        ax1.set_title(f'Raw EEG Signal - Channel {channel_idx+1}', fontsize=14)
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Preprocessed signal
        ax2.plot(preprocessed_data[channel_idx], color='green', alpha=0.7, linewidth=0.8)
        ax2.set_title(f'Preprocessed EEG Signal - Channel {channel_idx+1}', fontsize=14)
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def plot_frequency_spectrum(self, eeg_data, sampling_rate=256, channel_idx=0):
        """
        Plot frequency spectrum using FFT
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get signal for specific channel
        signal_data = eeg_data[channel_idx]
        
        # Compute FFT
        fft_vals = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
        
        # Take only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        # Plot FFT
        ax1.plot(fft_freq, fft_vals, color='purple', linewidth=1)
        ax1.set_xlim(0, 60)  # Focus on 0-60 Hz range
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('FFT - Frequency Spectrum')
        ax1.grid(True, alpha=0.3)
        
        # Highlight frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 50)
        }
        
        colors = ['gray', 'blue', 'green', 'orange', 'red']
        for (band, (low, high)), color in zip(bands.items(), colors):
            ax1.axvspan(low, high, alpha=0.2, color=color, label=band)
        
        ax1.legend()
        
        # Plot Power Spectral Density
        freqs, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=256)
        ax2.semilogy(freqs, psd, color='darkblue', linewidth=1)
        ax2.set_xlim(0, 60)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density (dB/Hz)')
        ax2.set_title('Power Spectral Density (PSD)')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Frequency Analysis - Channel {channel_idx+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_bandpower_comparison(self, bandpower_features, user_ids):
        """
        Plot bandpower comparison across users
        """
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, band in enumerate(bands):
            ax = axes[i]
            
            # Extract bandpower for this band across users
            band_data = []
            for user_features in bandpower_features:
                band_data.append(user_features[i])
            
            ax.bar(range(len(band_data)), band_data, color='steelblue', alpha=0.7)
            ax.set_xlabel('User ID')
            ax.set_ylabel('Power')
            ax.set_title(f'{band} Band Power')
            ax.set_xticks(range(len(band_data)))
            ax.set_xticklabels([f'U{i+1}' for i in range(len(band_data))])
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide last subplot
        axes[-1].set_visible(False)
        
        plt.suptitle('Bandpower Distribution Across Users', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names, importance_scores, title="Feature Importance"):
        """
        Plot feature importance from Random Forest or similar models
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[-20:]  # Top 20 features
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance_scores[sorted_idx]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_features)), sorted_importance, color='teal')
        
        # Customize the plot
        plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
            plt.text(val, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
    def plot_model_comparison(self, model_results):
        """
        Compare performance of different models
        """
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        scores = {}
        for metric in metrics:
            scores[metric] = [model_results[m].get(metric, 0) for m in models]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for metric, color in zip(metrics, colors):
            offset = width * multiplier
            bars = ax.bar(x + offset, scores[metric], width, label=metric.capitalize(), color=color)
            multiplier += 1
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix for authentication results
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_training_history(self, history):
        """
        Plot ANN training history (accuracy and loss curves)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', color='red', linewidth=2)
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss', color='purple', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('ANN Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curves(self, y_true, y_pred_prob, n_classes):
        """
        Plot ROC curves for multi-class classification
        """
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_prob[:, i]
            
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Each Class', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_authentication_result(self, is_authenticated, confidence, threshold, user_id):
        """
        Plot authentication result visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gauge chart for confidence
        colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        
        # Create gauge
        if confidence >= threshold:
            gauge_color = '#10ac84'  # Green
            status = "AUTHORIZED"
            status_color = 'green'
        else:
            gauge_color = '#ee5a24'  # Orange/Red
            status = "UNAUTHORIZED"
            status_color = 'red'
        
        # Left: Confidence meter
        ax1.barh([0], [confidence], color=gauge_color, alpha=0.8, height=0.3)
        ax1.barh([0], [1-confidence], left=[confidence], color='lightgray', alpha=0.5, height=0.3)
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_title(f'Authentication Confidence: {confidence:.3f}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add percentage text
        ax1.text(confidence/2, 0, f'{confidence:.1%}', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white' if confidence > 0.3 else 'black')
        
        # Right: Status
        ax2.text(0.5, 0.6, status, fontsize=24, fontweight='bold', 
                color=status_color, ha='center', transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f'User: {user_id}', fontsize=16, 
                ha='center', transform=ax2.transAxes)
        ax2.text(0.5, 0.3, f'Threshold: {threshold}', fontsize=12, 
                ha='center', transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title('Authentication Result', fontsize=14, fontweight='bold')
        
        plt.suptitle('EEG Authentication System - Live Result', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
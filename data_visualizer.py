import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import json

class DataVisualizer:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and validate data from various file formats."""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSON.")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def detect_anomalies(self, column, threshold=2):
        """Detect anomalies in numerical data using z-score method."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
        anomalies = self.data[z_scores > threshold]
        return anomalies

    def visualize_temporal_patterns(self, timestamp_col, value_col):
        """Visualize patterns over time, useful for analyzing digital evidence timestamps."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(self.data[timestamp_col]), self.data[value_col])
        plt.title(f"Temporal Analysis: {value_col} Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt

    def create_heatmap(self, data_matrix, title="Connection Strength Heatmap"):
        """Create heatmap visualization for showing relationships/connections."""
        plt.figure(figsize=(10, 8))
        plt.imshow(data_matrix, cmap='YlOrRd')
        plt.colorbar(label='Connection Strength')
        plt.title(title)
        plt.tight_layout()
        return plt

    def pattern_recognition(self, features, n_components=2):
        """Implement pattern recognition using PCA for dimensional reduction."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        X = self.data[features]
        X_scaled = self.scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.title("Pattern Recognition using PCA")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.tight_layout()
        return plt, pca.explained_variance_ratio_

    def generate_histogram(self, column, bins=30):
        """Generate histogram for frequency analysis."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column], bins=bins, edgecolor='black')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        return plt

    def save_visualization(self, plt_object, filename):
        """Save visualization with error handling."""
        try:
            plt_object.savefig(filename)
            print(f"Visualization saved successfully as {filename}")
            return True
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
            return False

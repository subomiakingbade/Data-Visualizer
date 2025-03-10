import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_visualizer import DataVisualizer

def generate_police_data():
    """Generate sample police-related data"""
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    
    # Simulate case processing data
    data = {
        'timestamp': dates,
        'case_processing_time': np.random.gamma(5, 1, 30),  # Processing times
        'evidence_count': np.random.poisson(10, 30),        # Evidence items per case
        'risk_score': np.random.uniform(0, 1, 30),         # Case risk assessment
        'suspect_count': np.random.poisson(2, 30)          # Suspects per case
    }
    
    df = pd.DataFrame(data)
    df.to_csv('police_case_data.csv', index=False)
    return df

def generate_general_data():
    """Generate sample general business data"""
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    
    # Simulate business metrics
    data = {
        'timestamp': dates,
        'sales': np.random.normal(1000, 200, 30),          # Daily sales
        'customer_count': np.random.poisson(100, 30),      # Daily customers
        'satisfaction_score': np.random.normal(4.2, 0.3, 30), # Customer satisfaction
        'processing_time': np.random.gamma(5, 1, 30)       # Order processing time
    }
    
    df = pd.DataFrame(data)
    df.to_csv('business_data.csv', index=False)
    return df

def main():
    viz = DataVisualizer()
    
    # Demo with police data
    print("Analyzing Police Case Data:")
    police_data = generate_police_data()
    viz.load_data('police_case_data.csv')
    
    # Temporal analysis of case processing
    plt = viz.visualize_temporal_patterns('timestamp', 'case_processing_time')
    plt.savefig('police_temporal_analysis.png')
    plt.close()
    
    # Pattern recognition in case features
    features = ['case_processing_time', 'evidence_count', 'risk_score', 'suspect_count']
    plt, variance_ratio = viz.pattern_recognition(features)
    plt.savefig('police_pattern_analysis.png')
    plt.close()
    print(f"Police data pattern analysis variance explained: {variance_ratio}")
    
    # Anomaly detection in processing times
    anomalies = viz.detect_anomalies('case_processing_time', threshold=2)
    print(f"Detected {len(anomalies)} anomalies in case processing times")
    
    # Demo with general business data
    print("\nAnalyzing Business Data:")
    business_data = generate_general_data()
    viz.load_data('business_data.csv')
    
    # Temporal analysis of sales
    plt = viz.visualize_temporal_patterns('timestamp', 'sales')
    plt.savefig('business_temporal_analysis.png')
    plt.close()
    
    # Pattern recognition in business metrics
    features = ['sales', 'customer_count', 'satisfaction_score', 'processing_time']
    plt, variance_ratio = viz.pattern_recognition(features)
    plt.savefig('business_pattern_analysis.png')
    plt.close()
    print(f"Business data pattern analysis variance explained: {variance_ratio}")
    
    # Anomaly detection in sales
    anomalies = viz.detect_anomalies('sales', threshold=2)
    print(f"Detected {len(anomalies)} anomalies in sales data")

if __name__ == "__main__":
    main()

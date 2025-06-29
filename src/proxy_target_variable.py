from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

def calculate_rfm(df, snaphot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features from a DataFrame.
    
    Parameters:
    df (DataFrame): Input DataFrame containing 'CustomerId', 'transaction_date', and 'amount'.
    snaphot_date (datetime, optional): The date to consider as the snapshot for recency calculation.
    
    Returns:
    DataFrame: A DataFrame with RFM features.
    """
    if snaphot_date is None:
        snaphot_date = df['TransactionStartTime'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snaphot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(
        columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        }
    ).reset_index()
    
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    
    return rfm

def scale_rfm(rfm_df):
    rfm_values = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_values)

    return scaled_rfm, scaler

def create_rfm_clusters(scaled_rfm, random_state=42):
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_rfm)
    return clusters

def assign_proxy_label(rfm_df, clusters):
    rfm_df['cluster'] = clusters

    # Compute mean values per cluster to identify least engaged one
    cluster_profile = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    print (cluster_profile)

    # Let's assume lowest frequency & monitary = high risk

    high_risk_cluster = cluster_profile.sort_values(by=['Frequency', 'Monetary']).index[0]
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

    return rfm_df

def save_labeled_data(df_labeled, file_path):
    """
    Save the labeled RFM DataFrame to a CSV file.
    
    Parameters:
    rfm_df (DataFrame): The DataFrame with RFM features and labels.
    file_path (str): The path where the DataFrame should be saved.
    """
    df_labeled.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('data/raw/data.csv')
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Calculate RFM
    rfm = calculate_rfm(df)

    # Scale RFM
    scale_rfm, scaler = scale_rfm(rfm)

    # Create clusters
    clusters = create_rfm_clusters(scale_rfm)

    # Assign proxy label
    rfm_labeled = assign_proxy_label(rfm, clusters)
    
    # Merge with original DataFrame
    df_labeled = df.merge(rfm_labeled[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

    # Save labeled data
    save_labeled_data(df_labeled, 'data/processed/labeled_data.csv')
    
    print(df_labeled.sample(5))
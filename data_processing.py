import pandas as pd
from models import KMEANS

def filter_columns(data):
    #takes in dataframe, returns with processed columns 
    data = data[ data['security_id'] == 1]
    data.drop('row_id', axis=1, inplace=True)
    data.drop('security_id', axis=1, inplace=True)
    data.drop('initiator', axis=1, inplace=True)

    for i in range(1, 51):
        data.drop(f'time{i}', axis=1, inplace=True)
        data.drop(f'transtype{i}', axis=1, inplace=True)
    
    for i in range(61, 101):
        data.drop(f'bid{i}', axis=1, inplace=True)
        data.drop(f'ask{i}', axis=1, inplace=True)

    return data

def scale_data(data): 
    scaled_data = (data - data.mean()) / data.std()
    return scaled_data

def split_data(data):
    #returns matrices X, y
    target_columns = []
    for i in range(51, 61):
        target_columns.append(f'bid{i}')
        target_columns.append(f'ask{i}')
    
    X = data.drop(columns=target_columns)
    y = data[target_columns]

    return X, y


# load and process data
def get_matrices(df):
    data = filter_columns(df)
    scaled_data = scale_data(data)
    X, y = split_data(scaled_data)
    return X, y
        

# apply k-means clusters, returns augmented df 
def apply_kmeans_labels(df, y_vector):
    kmeans_model = KMEANS(n_clusters=5)
    clusters = kmeans_model.fit(y_vector)
    df['cluster_label'] = clusters
    return df


        
import torch
import torch.nn as nn
import torch.optim as optim 
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC


class KMEANS:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.labels_ = None 

    def fit(self, y):
        if hasattr(y, 'values'):
            y = y.values
        
        self.model.fit(y)
        self.labels_ = self.model.labels_
        return self.labels_
    
    def predict(self, y):
        if hasattr(y, 'values'):
            y = y.values
        return self.model.predict(y)

    def fit_predict(self, y):
        return self.fit(y)

    def get_centroids(self):
        return self.model.cluster_centers_
    


class NaiveRepeaterModel:
    def __init__(self, output_length=20):
        self.output_length = output_length
        self.last_two = None
    
    def fit(self, X, y=None):
        self.last_two = X[:, -2:]
    
    def predict(self, X):
        #num_samples = X.shape[0]
        repeats = self.output_length // 2
        preds = np.tile(self.last_two, (1, repeats))
        return preds


class MultiOutputRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    
    def forward(self, x):
        return self.linear(x)
    

class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.shared_cov = None
        self.classes = []
        self.inv_shared_cov = None
        self.const_term = None

    def fit(self, X, y):
        """
        X: torch.Tensor of shape (n_samples, n_features)
        y: torch.Tensor of shape (n_samples,)
        """
        self.classes = torch.unique(y).tolist()
        n_samples, n_features = X.shape

        self.class_priors = {}
        self.class_means = {}

        cov_sum = torch.zeros((n_features, n_features))

        for cls in self.classes:
            X_k = X[y == cls]
            n_k = X_k.shape[0]

            self.class_priors[cls] = n_k / n_samples
            self.class_means[cls] = X_k.mean(dim=0)

            centered = X_k - self.class_means[cls]
            cov_sum += centered.T @ centered

        self.shared_cov = cov_sum / n_samples
        self.inv_shared_cov = torch.inverse(self.shared_cov)
        self.const_term = -0.5 * n_features * torch.log(torch.tensor(2 * np.pi)) - 0.5 * torch.logdet(self.shared_cov)

    def predict(self, x):
        """
        x: torch.Tensor of shape (n_features,) or (1, n_features)
        returns: int (predicted class)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        scores = []

        for cls in self.classes:
            mu_k = self.class_means[cls]
            prior_k = self.class_priors[cls]

            diff = x - mu_k
            log_likelihood = -0.5 * (diff @ self.inv_shared_cov @ diff.T).squeeze()
            log_posterior = self.const_term + log_likelihood + torch.log(torch.tensor(prior_k))

            scores.append(log_posterior)

        return torch.argmax(torch.tensor(scores)).item()

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)
        self.input_cols = None

    def fit(self, X, y, feature_cols=None):
        if isinstance(X, pd.DataFrame):
            if feature_cols is None:
                feature_cols = X.columns.tolist()
            self.input_cols = feature_cols
            X_vals = X[feature_cols].values
        else:
            X_vals = X  # assume already a NumPy array

        self.model.fit(X_vals, y)

    def predict(self, row):
        if self.input_cols is None:
            raise ValueError("Model has not been trained yet.")
        x = row[self.input_cols].values.reshape(1, -1)
        return self.model.predict(x)[0]

    def predict_batch(self, df):
        if self.input_cols is None:
            raise ValueError("Model has not been trained yet.")
        X = df[self.input_cols].values
        return self.model.predict(X)


class TransformerRegressor(nn.Module):
    def __init__(self, timestep_dim=2, seq_len=50, aux_dim=4, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=20):
        super().__init__()

        self.seq_len = seq_len
        self.timestep_dim = timestep_dim
        self.aux_dim = aux_dim

        self.input_proj = nn.Linear(timestep_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model + aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

            # Split sequence and aux features
        seq = x[:, :self.seq_len * self.timestep_dim]
        aux = x[:, self.seq_len * self.timestep_dim:]

        seq = seq.view(batch_size, self.seq_len, self.timestep_dim)
        seq = self.input_proj(seq) + self.positional_encoding.unsqueeze(0)

        z = self.transformer(seq)
        pooled = z.mean(dim=1)

        combined = torch.cat([pooled, aux], dim=1)
        return self.output_layer(combined)
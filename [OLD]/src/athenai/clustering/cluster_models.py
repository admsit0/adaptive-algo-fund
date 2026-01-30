"""
Clustering models for algo grouping.

Provides:
- MiniBatchKMeansModel: Fast, scalable K-Means
- KMeansModel: Standard K-Means++ 
- GMMModel: Gaussian Mixture Model

All models implement:
- fit(X): Train on feature matrix
- predict(X): Assign cluster labels
- save(path): Save model to disk
- load(path): Load model from disk
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class ClusterModel(ABC):
    """Abstract base class for clustering models."""
    
    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.seed = seed
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.metadata: dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "ClusterModel":
        """Fit the model to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return labels."""
        self.fit(X)
        return self.labels_
    
    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> "ClusterModel":
        """Load model from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_distances(self, X: np.ndarray) -> np.ndarray:
        """Get distance to assigned cluster center."""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted")
        labels = self.predict(X)
        distances = np.zeros(len(X))
        for i, (x, label) in enumerate(zip(X, labels)):
            distances[i] = np.linalg.norm(x - self.cluster_centers_[label])
        return distances


class MiniBatchKMeansModel(ClusterModel):
    """
    Mini-Batch K-Means clustering.
    
    Fast and scalable, good for large datasets.
    Uses random mini-batches to update centroids.
    """
    
    def __init__(
        self,
        k: int,
        seed: int = 42,
        batch_size: int = 1024,
        max_iter: int = 100,
        n_init: int = 3,
        tol: float = 1e-4,
    ):
        super().__init__(k, seed)
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
    
    def fit(self, X: np.ndarray) -> "MiniBatchKMeansModel":
        """Fit using mini-batch k-means."""
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X.shape
        
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        for init_idx in range(self.n_init):
            # K-means++ initialization
            centers = self._kmeans_plusplus_init(X, rng)
            
            # Mini-batch iterations
            counts = np.zeros(self.k)
            
            for iteration in range(self.max_iter):
                # Sample mini-batch
                batch_indices = rng.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
                X_batch = X[batch_indices]
                
                # Assign to nearest center
                distances = self._compute_distances(X_batch, centers)
                labels_batch = np.argmin(distances, axis=1)
                
                # Update centers
                old_centers = centers.copy()
                for j in range(self.k):
                    mask = labels_batch == j
                    if np.any(mask):
                        counts[j] += mask.sum()
                        eta = 1.0 / counts[j]
                        centers[j] = (1 - eta) * centers[j] + eta * X_batch[mask].mean(axis=0)
                
                # Check convergence
                center_shift = np.linalg.norm(centers - old_centers)
                if center_shift < self.tol:
                    break
            
            # Compute final inertia
            distances = self._compute_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            inertia = np.sum(np.min(distances, axis=1) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.metadata["algorithm"] = "minibatch_kmeans"
        self.metadata["n_iter"] = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted")
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def _kmeans_plusplus_init(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-means++ initialization."""
        n_samples = X.shape[0]
        centers = np.zeros((self.k, X.shape[1]))
        
        # First center: random
        idx = rng.randint(n_samples)
        centers[0] = X[idx]
        
        # Remaining centers: proportional to squared distance
        for i in range(1, self.k):
            distances = self._compute_distances(X, centers[:i])
            min_distances = np.min(distances, axis=1)
            probs = min_distances ** 2
            probs /= probs.sum()
            idx = rng.choice(n_samples, p=probs)
            centers[i] = X[idx]
        
        return centers
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute distances from X to centers."""
        # Shape: (n_samples, k)
        return np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))


class KMeansModel(ClusterModel):
    """
    Standard K-Means++ clustering.
    
    Full batch updates, more accurate but slower than mini-batch.
    """
    
    def __init__(
        self,
        k: int,
        seed: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
    ):
        super().__init__(k, seed)
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
    
    def fit(self, X: np.ndarray) -> "KMeansModel":
        """Fit using standard k-means."""
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]
        
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        for init_idx in range(self.n_init):
            # K-means++ initialization
            centers = self._kmeans_plusplus_init(X, rng)
            
            for iteration in range(self.max_iter):
                # Assign to nearest center
                distances = self._compute_distances(X, centers)
                labels = np.argmin(distances, axis=1)
                
                # Update centers
                old_centers = centers.copy()
                for j in range(self.k):
                    mask = labels == j
                    if np.any(mask):
                        centers[j] = X[mask].mean(axis=0)
                
                # Check convergence
                center_shift = np.linalg.norm(centers - old_centers)
                if center_shift < self.tol:
                    break
            
            # Compute inertia
            distances = self._compute_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            inertia = np.sum(np.min(distances, axis=1) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.metadata["algorithm"] = "kmeans"
        self.metadata["n_iter"] = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted")
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def _kmeans_plusplus_init(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-means++ initialization."""
        n_samples = X.shape[0]
        centers = np.zeros((self.k, X.shape[1]))
        
        idx = rng.randint(n_samples)
        centers[0] = X[idx]
        
        for i in range(1, self.k):
            distances = self._compute_distances(X, centers[:i])
            min_distances = np.min(distances, axis=1)
            probs = min_distances ** 2
            probs /= probs.sum()
            idx = rng.choice(n_samples, p=probs)
            centers[i] = X[idx]
        
        return centers
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute distances from X to centers."""
        return np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))


class GMMModel(ClusterModel):
    """
    Gaussian Mixture Model clustering.
    
    Soft clustering with probabilistic assignments.
    Uses EM algorithm with diagonal covariance.
    """
    
    def __init__(
        self,
        k: int,
        seed: int = 42,
        max_iter: int = 100,
        n_init: int = 3,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
    ):
        super().__init__(k, seed)
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.reg_covar = reg_covar
        
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
    
    def fit(self, X: np.ndarray) -> "GMMModel":
        """Fit using EM algorithm."""
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X.shape
        
        best_ll = -np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Initialize with k-means
            kmeans = KMeansModel(self.k, seed=self.seed + init_idx, n_init=1)
            kmeans.fit(X)
            
            means = kmeans.cluster_centers_.copy()
            weights = np.ones(self.k) / self.k
            covariances = np.ones((self.k, n_features)) + self.reg_covar
            
            for iteration in range(self.max_iter):
                # E-step: compute responsibilities
                log_resp = self._compute_log_responsibilities(X, weights, means, covariances)
                resp = np.exp(log_resp)
                
                # M-step: update parameters
                nk = resp.sum(axis=0) + 1e-10
                
                old_means = means.copy()
                weights = nk / n_samples
                means = (resp.T @ X) / nk[:, np.newaxis]
                
                # Update diagonal covariances
                for j in range(self.k):
                    diff = X - means[j]
                    covariances[j] = (resp[:, j:j+1] * diff ** 2).sum(axis=0) / nk[j]
                    covariances[j] += self.reg_covar
                
                # Check convergence
                mean_shift = np.linalg.norm(means - old_means)
                if mean_shift < self.tol:
                    break
            
            # Compute log-likelihood
            ll = self._compute_log_likelihood(X, weights, means, covariances)
            
            if ll > best_ll:
                best_ll = ll
                best_params = (weights.copy(), means.copy(), covariances.copy())
        
        self.weights_, self.means_, self.covariances_ = best_params
        self.cluster_centers_ = self.means_
        self.log_likelihood_ = best_ll
        
        # Assign labels
        log_resp = self._compute_log_responsibilities(X, self.weights_, self.means_, self.covariances_)
        self.labels_ = np.argmax(log_resp, axis=1)
        
        self.metadata["algorithm"] = "gmm"
        self.metadata["log_likelihood"] = best_ll
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels (hard assignment)."""
        if self.means_ is None:
            raise ValueError("Model not fitted")
        log_resp = self._compute_log_responsibilities(X, self.weights_, self.means_, self.covariances_)
        return np.argmax(log_resp, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities (soft assignment)."""
        if self.means_ is None:
            raise ValueError("Model not fitted")
        log_resp = self._compute_log_responsibilities(X, self.weights_, self.means_, self.covariances_)
        return np.exp(log_resp)
    
    def _compute_log_responsibilities(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        """Compute log responsibilities."""
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.k))
        
        for j in range(self.k):
            # Log probability under diagonal Gaussian
            diff = X - means[j]
            log_prob = -0.5 * np.sum(
                np.log(2 * np.pi * covariances[j]) + diff ** 2 / covariances[j],
                axis=1
            )
            log_resp[:, j] = np.log(weights[j] + 1e-10) + log_prob
        
        # Normalize
        log_sum = np.log(np.exp(log_resp).sum(axis=1, keepdims=True) + 1e-10)
        log_resp -= log_sum
        
        return log_resp
    
    def _compute_log_likelihood(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> float:
        """Compute log likelihood."""
        n_samples = X.shape[0]
        ll = 0.0
        
        for i in range(n_samples):
            sample_ll = 0.0
            for j in range(self.k):
                diff = X[i] - means[j]
                log_prob = -0.5 * np.sum(
                    np.log(2 * np.pi * covariances[j]) + diff ** 2 / covariances[j]
                )
                sample_ll += weights[j] * np.exp(log_prob)
            ll += np.log(sample_ll + 1e-10)
        
        return ll


def get_cluster_model(algorithm: str, k: int, seed: int = 42, **kwargs) -> ClusterModel:
    """Factory function to get clustering model."""
    if algorithm == "minibatch_kmeans":
        return MiniBatchKMeansModel(k=k, seed=seed, **kwargs)
    elif algorithm == "kmeans":
        return KMeansModel(k=k, seed=seed, **kwargs)
    elif algorithm == "gmm":
        return GMMModel(k=k, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class RobustScaler:
    """Robust scaler using median and IQR."""
    
    def __init__(self, clip_range: float = 5.0):
        self.clip_range = clip_range
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> "RobustScaler":
        """Fit scaler to data."""
        self.median_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ < 1e-10] = 1.0  # Avoid division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.median_ is None:
            raise ValueError("Scaler not fitted")
        X_scaled = (X - self.median_) / self.iqr_
        # Clip outliers
        X_scaled = np.clip(X_scaled, -self.clip_range, self.clip_range)
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class ZScoreScaler:
    """Standard z-score scaler with optional clipping."""
    
    def __init__(self, clip_range: float = 5.0):
        self.clip_range = clip_range
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        """Fit scaler to data."""
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.mean_ is None:
            raise ValueError("Scaler not fitted")
        X_scaled = (X - self.mean_) / self.std_
        X_scaled = np.clip(X_scaled, -self.clip_range, self.clip_range)
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


def get_scaler(scaler_type: str, **kwargs):
    """Factory function to get scaler."""
    if scaler_type == "robust":
        return RobustScaler(**kwargs)
    elif scaler_type == "zscore":
        return ZScoreScaler(**kwargs)
    elif scaler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")

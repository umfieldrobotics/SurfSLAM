from common.settings import Settings
import faiss
import torch
from typing import List, Tuple
import numpy as np


class ImageDescriptorManager:    
    def __init__(self, 
                 settings: Settings,
                 device: torch.device = 'cuda:0'):
        self.n_clusters = settings.n_clusters
        self.descriptor_dim = settings.descriptor_dim
        self.device = torch.device(device)
        self.embedding_dim = self.n_clusters * self.descriptor_dim
        
        self.cluster_centers = None
        self.faiss_index = None
        self.is_fitted = False
                    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search."""
        if self.device is not None and self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatL2(res, self.descriptor_dim)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.descriptor_dim)
            
        self.faiss_index.add(self.cluster_centers)
                    
    def fit(self, descriptors_list: List[np.ndarray], max_descriptors: int = 50000):
        """Fit cluster centers for VLAD."""
        all_desc = np.vstack(descriptors_list)
        
        if len(all_desc) > max_descriptors:
            indices = np.random.choice(len(all_desc), max_descriptors, replace=False)
            all_desc = all_desc[indices]
            
        all_desc = np.ascontiguousarray(all_desc.astype(np.float32))
        use_gpu = self.device is not None and self.device.type == 'cuda'
        
        print(f"Fitting VLAD with {len(all_desc)} descriptors, {self.n_clusters} clusters (GPU={use_gpu})...")
        max_points = int(len(all_desc) / self.n_clusters * 1.5)  # Add some headroom

        kmeans = faiss.Kmeans(
            d=self.descriptor_dim,
            k=self.n_clusters,
            niter=500,
            nredo=3,
            verbose=True,
            gpu=use_gpu,
            seed=42,
            max_points_per_centroid=max_points
        )
        kmeans.train(all_desc)
        self.cluster_centers = np.ascontiguousarray(
            kmeans.centroids.astype(np.float32)
        )
        
        self._build_faiss_index()
        self.is_fitted = True
        print(f"  Cluster centers shape: {self.cluster_centers.shape}")
        
    def _assign_clusters(self, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign descriptors to nearest clusters using FAISS."""
        descriptors = np.ascontiguousarray(descriptors.astype(np.float32))
        distances, assignments = self.faiss_index.search(descriptors, 1)
        return assignments.flatten(), distances.flatten()
    
    def aggregate(self,
                  descriptors: np.ndarray | torch.Tensor,
                  scores: np.ndarray | torch.Tensor | None = None) -> np.ndarray | torch.Tensor:
        """Aggregate local descriptors into a VLAD global descriptor."""
        is_torch = isinstance(descriptors, torch.Tensor)

        if len(descriptors) == 0:
            if is_torch:
                return torch.zeros(self.embedding_dim, dtype=torch.float32, device=descriptors.device)
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if not self.is_fitted:
            raise RuntimeError("VLAD requires fitting. Call fit() first.")

        # Convert torch to numpy for FAISS assignment, preserving device
        if is_torch:
            device = descriptors.device
            descriptors_np = descriptors.detach().cpu().numpy().astype(np.float32)
            if scores is not None:
                scores_np = scores.detach().cpu().numpy()
            else:
                scores_np = None
        else:
            descriptors_np = descriptors.astype(np.float32)
            scores_np = scores

        # Compute weights
        if scores_np is not None:
            weights_np = scores_np / (scores_np.sum() + 1e-8)
        else:
            weights_np = np.ones(len(descriptors_np), dtype=np.float32) / len(descriptors_np)

        # Assign clusters using FAISS (always numpy)
        assignments, _ = self._assign_clusters(descriptors_np)

        if is_torch:
            # Torch implementation
            descriptors_t = torch.from_numpy(descriptors_np).to(device)
            weights_t = torch.from_numpy(weights_np).to(device)
            assignments_t = torch.from_numpy(assignments).to(device)
            centers_t = torch.from_numpy(self.cluster_centers).to(device)

            vlad = torch.zeros((self.n_clusters, self.descriptor_dim),
                             dtype=torch.float32, device=device)

            # Vectorized residual computation
            assigned_centers = centers_t[assignments_t]
            residuals = descriptors_t - assigned_centers

            # Weighted accumulation per cluster
            weighted_residuals = weights_t[:, None] * residuals
            vlad.index_add_(0, assignments_t, weighted_residuals)

            # Intra-normalization
            norms = torch.norm(vlad, dim=1, keepdim=True)
            vlad = torch.where(norms > 1e-8, vlad / norms, vlad)

            embedding = vlad.flatten()

            # L2 normalize
            norm = torch.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        else:
            # Numpy implementation
            vlad = np.zeros((self.n_clusters, self.descriptor_dim), dtype=np.float32)

            # Vectorized residual computation
            assigned_centers = self.cluster_centers[assignments]
            residuals = descriptors_np - assigned_centers

            # Weighted accumulation per cluster
            np.add.at(vlad, assignments, weights_np[:, None] * residuals)

            # Intra-normalization
            norms = np.linalg.norm(vlad, axis=1, keepdims=True)
            vlad = np.divide(vlad, norms, where=norms > 1e-8, out=vlad)

            embedding = vlad.flatten()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding /= norm

        return embedding
    
    def compute_similarity(self,
                           embedding1: np.ndarray | torch.Tensor,
                           embedding2: np.ndarray | torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        if isinstance(embedding1, torch.Tensor):
            return float(torch.dot(embedding1, embedding2))
        return float(np.dot(embedding1, embedding2))
    
    def compute_similarity_batch(self, 
                                 query: np.ndarray | torch.Tensor, 
                                 database: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Compute cosine similarities between query and all database embeddings."""
        return database @ query
    
    def save(self, path: str):
        """Save fitted state to a file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted descriptor.")
            
        np.savez(path,
                 n_clusters=self.n_clusters,
                 descriptor_dim=self.descriptor_dim,
                 embedding_dim=self.embedding_dim,
                 cluster_centers=self.cluster_centers)
        print(f"Saved descriptor state to {path}")
        
    def load(self, path: str):
        """Load fitted state from a file."""
        data = np.load(path)
        
        self.n_clusters = int(data['n_clusters'])
        self.descriptor_dim = int(data['descriptor_dim'])
        self.embedding_dim = int(data['embedding_dim'])
        self.cluster_centers = np.ascontiguousarray(data['cluster_centers'])
        
        self._build_faiss_index()
        self.is_fitted = True
            
        print(f"Loaded descriptor state from {path}")
    
    @classmethod
    def from_file(cls, path: str, settings: Settings = None, device: torch.device = 'cuda:0') -> 'ImageDescriptorManager':
        """Create a new instance from a saved file."""
        data = np.load(path)
        
        if settings is None:
            class MinimalSettings:
                pass
            settings = MinimalSettings()
            
        settings.n_clusters = int(data['n_clusters'])
        settings.descriptor_dim = int(data['descriptor_dim'])
        
        instance = cls(settings, device)
        instance.load(path)
        return instance
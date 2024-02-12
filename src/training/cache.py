import torch


class FeatureCache:
    """FIFO queue for caching feature activations"""

    def __init__(
        self, 
        cache_size: int, 
        dict_size: int, 
        dtype: torch.dtype = torch.float32
    ):
        """
        :param cache_size: Size of the FeatureCache.
        :param dict_size: Hidden dimension of the sparse autoencoder.
        """
        self.cache = torch.zeros((cache_size, dict_size), dtype=dtype)
        
    def push(self, feature_activations: torch.Tensor):
        assert feature_activations.shape == (1, self.cache.size(1))
        self.cache = torch.cat((self.cache[1:], feature_activations))
        
    def get(self):
        return self.cache

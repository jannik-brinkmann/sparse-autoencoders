import torch


class ActivationBuffer:

    def __init__(
        self, 
        n_samples: int, 
        n_features: int, 
        dtype: torch.dtype = torch.float32
    ):
        """
        initialize the ActivationBuffer.

        args:
        n_samples (int): The number of samples in the ActivationBuffer.
        n_features (int): The number of features in the sparse autoencoder. 
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.buffer = torch.empty((n_samples, n_features), dtype=dtype)
        self.current_size = 0
        self.full = False

    def push(self, batch: torch.Tensor):
        """
        push a new batch into the ActivationBuffer.

        args:
        batch (Tensor): The new batch to be added.
        """
        batch_size = batch.shape[0]

        if self.current_size + batch_size <= self.n_samples:
            # If buffer has enough space, append the batch
            self.buffer[self.current_size:self.current_size + batch_size] = batch
            self.current_size += batch_size
        # else:
        #     # Calculate the space needed and shift elements
        #     shift = self.current_size + batch_size - self.n_samples
        #     self.buffer[:self.current_size - shift] = self.buffer[shift:self.current_size].clone()
        #     self.buffer[self.current_size - shift:self.n_samples] = batch
        #     self.current_size = self.n_samples
        else: # loop back to the beginning (& ignore the unused end space)
            # Overwrite the first batch_size elements
            self.buffer[:batch_size] = batch
            self.current_size = batch_size
            self.full = True

    def get(self):
        """
        get the current elements of the ActivationBuffer.
        """
        return self.buffer[:self.current_size]

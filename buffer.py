import torch

class TensorBuffer:
    def __init__(self, n_samples, n_features, dtype=torch.float32):
        """
        Initialize the TensorBuffer.

        Args:
        max_size (int): The maximum number of batches in the buffer.
        shape (tuple): The shape of each individual batch.
        dtype: The data type of the elements.
        """
        self.n_samples = n_samples
        self.buffer_shape = (n_samples, n_features)
        self.buffer = torch.empty(self.buffer_shape, dtype=dtype)
        self.current_size = 0

    def push(self, batch):
        """
        Push a new batch into the buffer.

        Args:
        batch (Tensor): The new batch to be added.
        """
        batch_size = batch.shape[0]

        if self.current_size + batch_size <= self.n_samples:
            # If buffer has enough space, append the batch
            self.buffer[self.current_size:self.current_size + batch_size] = batch
            self.current_size += batch_size
        else:
            # Calculate the space needed and shift elements
            shift = self.current_size + batch_size - self.n_samples
            self.buffer[:self.current_size - shift] = self.buffer[shift:self.current_size].clone()
            self.buffer[self.current_size - shift:self.current_size] = batch
            self.current_size = self.n_samples

    def get(self):
        """
        Get the current elements of the buffer.
        """
        return self.buffer[:self.current_size]


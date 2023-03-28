import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA1D(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Modified 1D version
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(self, embedding_dim, num_embeddings, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def encode(self, x):
        """
        Input the vector to be quantized and output the codebook indices
        """
        # [B, D, T] -> [B, T, D]
        x = x.permute(0, 2, 1).contiguous()
        # [B, T, D] -> [B * T, D]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        encoding_indices = encoding_indices.reshape(x.shape[0], x.shape[1])

        # [B, T,]
        return encoding_indices
    
    def decode(self, indices):
        """
        Input codebook indices, output quantized vectors.
        """
        quantized = self.quantize(indices)
        quantized = quantized.permute(0, 2, 1).contiguous()

        return quantized
    
    def forward(self, x):
        """
        Input shape:[B, D, T]
        Output shape:[B, D, T], vq_loss
        D indicates the features of each frame
        """
        # [B, D, T] -> [B, T, D]
        x = x.permute(0, 2, 1).contiguous()
        # [B, T, D] -> [B * T, D]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, T, D]

        if not self.training:
            # e_latent_loss = F.mse_loss(x, quantized.detach())
            quantized = quantized.permute(0, 2, 1).contiguous()
            # loss = self.commitment_cost * e_latent_loss
            return quantized, quantized.detach()
            # return quantized, loss

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        # e_latent_loss = F.mse_loss(x, quantized.detach())
        # loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 2, 1).contiguous()
        return quantized, quantized.detach()
        # return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
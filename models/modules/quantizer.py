import torch
import torch.nn as nn
import torch.nn.functional as F


"""
The standard VQ implementation, where the embedding is initialized using 
nn.init.xavier_uniform_ and EMA is in the form of bias correction.
"""


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
        average = self.hidden / (1 - self.decay**self.counter)  # bias correction
        return average


class VectorQuantizerEMA1D(nn.Module):
    """VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Modified 1D version
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss
        terms (see equation 4 in the paper - this variable is Beta).
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
        self.ema_cluster_size = ExponentialMovingAverage(
            torch.zeros((self.num_embeddings,)), decay
        )

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
            updated_ema_cluster_size = self.ema_cluster_size(
                torch.sum(encodings, dim=0)
            )
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            dw = torch.matmul(
                encodings.t(), flat_x
            )  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            )
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
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2.0 * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


class VectorQuantizerEMAProject1D(nn.Module):
    """VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Modified 1D version
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss
        terms (see equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(
        self, embedding_dim, project_dim, num_embeddings, decay=0.99, epsilon=1e-5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        self.project_dim = project_dim
        self.proj_in = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.project_dim, kernel_size=1),
            nn.BatchNorm1d(self.project_dim),
        )
        self.proj_out = nn.Sequential(
            nn.Conv1d(self.project_dim, self.embedding_dim, kernel_size=1),
            nn.BatchNorm1d(self.embedding_dim),
        )

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.project_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(
            torch.zeros((self.num_embeddings,)), decay
        )

    def encode(self, x):
        """
        Input the vector to be quantized and output the codebook indices
        """
        x = self.proj_in(x)
        # [B, D, T] -> [B, T, D]
        x = x.permute(0, 2, 1).contiguous()
        # [B, T, D] -> [B * T, D]
        flat_x = x.reshape(-1, self.project_dim)

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

        return self.proj_out(quantized)

    def forward(self, x):
        """
        Input shape:[B, D, T]
        Output shape:[B, D, T], vq_loss
        D indicates the features of each frame
        """
        # [B, D, T] -> [B, T, D]
        x = self.proj_in(x)
        x = x.permute(0, 2, 1).contiguous()
        # [B, T, D] -> [B * T, D]
        flat_x = x.reshape(-1, self.project_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, T, D]

        if not self.training:
            # e_latent_loss = F.mse_loss(x, quantized.detach())
            quantized = quantized.permute(0, 2, 1).contiguous()
            # loss = self.commitment_cost * e_latent_loss
            x = x.permute(0, 2, 1).contiguous()
            return self.proj_out(quantized), x, quantized.detach()
            # return quantized, loss

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(
                torch.sum(encodings, dim=0)
            )
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            dw = torch.matmul(
                encodings.t(), flat_x
            )  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            )
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        # e_latent_loss = F.mse_loss(x, quantized.detach())
        # loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 2, 1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        return self.proj_out(quantized), x, quantized.detach()
        # return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2.0 * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, embedding_dim, project_dim, num_quantizers, vq_byte, decay=0.99):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_embeddings = 2**vq_byte
        self.num_quantizers = num_quantizers
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizerEMAProject1D(
                    self.embedding_dim,
                    self.project_dim,
                    self.num_embeddings,
                    decay=decay,
                )
                for _ in range(self.num_quantizers)
            ]
        )

    def encode(self, x):
        residual = x
        indices_list = []
        n_q = self.num_quantizers

        for layer in self.codebooks[:n_q]:
            indices = layer.encode(residual)
            quantized, _, _ = layer(residual)
            residual = residual - quantized.detach()
            # considering only the first layer's gradient
            indices_list.append(indices)

        return indices_list

    def decode(self, indices_list):
        quantized_out = 0.0

        for idx, indices in enumerate(indices_list):
            quantized = self.codebooks[idx].decode(indices)
            quantized_out = quantized_out + quantized

        return quantized_out

    def forward(self, x):
        quantized_out = 0.0
        residual = x

        input_list = []
        q_detach_list = []

        n_q = self.num_quantizers
        # no dropout

        for layer in self.codebooks[:n_q]:
            quantized, vq_in, vq_out = layer(residual)
            residual = residual - quantized
            # considering only the first layer's gradient
            quantized_out = quantized_out + quantized

            input_list.append(vq_in)
            q_detach_list.append(vq_out)

        return quantized_out, input_list, q_detach_list


class GroupVectorQuantizer(nn.Module):
    """Group VectorQuantizer by using VectorQuantizerEMA1D.
    Flatten the features of each frame, then group these
    features and perform vector quantization.
    """

    def __init__(
        self, embedding_dim, num_groups, vq_byte, project_dim=None, decay=0.99
    ):
        super(GroupVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        assert embedding_dim % num_groups == 0, (
            f"Can't group by the setting: embedding_dim={embedding_dim}, "
            f"num_groups={num_groups} "
        )

        self.sub_dim = embedding_dim // num_groups
        self.num_embeddings = 2**vq_byte
        # for each codebook

        if project_dim:
            self.project_dim = project_dim
            self.codebooks = nn.ModuleList(
                [
                    VectorQuantizerEMAProject1D(
                        self.sub_dim, self.project_dim, self.num_embeddings, decay=decay
                    )
                    for _ in range(self.num_groups)
                ]
            )

        else:
            self.codebooks = nn.ModuleList(
                [
                    VectorQuantizerEMA1D(self.sub_dim, self.num_embeddings, decay=decay)
                    for _ in range(self.num_groups)
                ]
            )

    def encode(self, x):
        indices_list = []

        for idx, s in enumerate(torch.chunk(x, self.num_groups, dim=1)):
            indices = self.codebooks[idx].encode(s)
            # [B, T,]
            indices_list.append(indices)

        return indices_list

    def decode(self, indices_list):
        s_q_list = []

        for idx, indices in enumerate(indices_list):
            s_q = self.codebooks[idx].decode(indices)
            s_q_list.append(s_q)

        x_q = torch.cat(s_q_list, dim=1)

        return x_q

    def forward(self, x):
        """
        Input shape:[B, D, T]
        Output shape:[B, D, T], and input_list, s_q_detach_list for commitment loss
        """

        input_list = []
        s_q_list = []
        s_q_detach_list = []
        for idx, s in enumerate(torch.chunk(x, self.num_groups, dim=1)):
            # [B, sub_dim, T]
            if self.project_dim:
                s_q, vq_in, vq_out = self.codebooks[idx](s)
                input_list.append(vq_in)
                s_q_list.append(s_q)
                s_q_detach_list.append(vq_out)
            else:
                s_q, s_q_detach = self.codebooks[idx](s)
                input_list.append(s)
                s_q_list.append(s_q)
                s_q_detach_list.append(s_q_detach)
        x_q = torch.cat(s_q_list, dim=1)

        return x_q, input_list, s_q_detach_list

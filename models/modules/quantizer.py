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


class VectorQuantize(nn.Module):
    def __init__(
        self, embedding_dim, project_dim, num_embeddings, decay=0.99, epsilon=1e-5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(
            torch.zeros((self.num_embeddings,)), decay
        )

        self.project_dim = project_dim
        if self.project_dim:
            self.proj_in = nn.Sequential(
                nn.Conv1d(self.embedding_dim, self.project_dim, kernel_size=1),
                nn.BatchNorm1d(self.project_dim),
            )
            self.proj_out = nn.Sequential(
                nn.Conv1d(self.project_dim, self.embedding_dim, kernel_size=1),
                nn.BatchNorm1d(self.embedding_dim),
            )
            embeddings = torch.empty(self.num_embeddings, self.project_dim)
        else: 
            # initialize embeddings as buffers
            embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

    def encode(self, x):
        """
        Input the vector to be quantized and output the codebook indices
        """
        if self.project_dim:
            vq_in = self.proj_in(x).permute(0, 2, 1).contiguous()
            flatten = vq_in.reshape(-1, self.project_dim)
        else:
            vq_in = x.permute(0, 2, 1).contiguous()
            flatten = vq_in.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flatten)
        encoding_indices = encoding_indices.reshape(vq_in.shape[0], vq_in.shape[1])

        # [B, T,]
        return encoding_indices

    def decode(self, indices):
        """
        Input codebook indices, output quantized vectors.
        """
        vq_out = self.quantize(indices)
        vq_out = vq_out.permute(0, 2, 1).contiguous()

        if self.project_dim:
            quantized = self.proj_out(vq_out)
        else:
            quantized = vq_out

        return quantized

    def forward(self, x):
        """
        Input shape:[B, D, T]
        Output shape:[B, D, T], vq_loss
        D indicates the features of each frame
        """
        if self.project_dim:
            vq_in = self.proj_in(x).permute(0, 2, 1).contiguous()
            flatten = vq_in.reshape(-1, self.project_dim)
        else:
            vq_in = x.permute(0, 2, 1).contiguous()
            flatten = vq_in.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flatten)
        vq_out = self.quantize(encoding_indices)
        vq_out = vq_out.view_as(vq_in)  # [B, T, D]

        if not self.training:
            vq_in = vq_in.permute(0, 2, 1).contiguous()
            vq_out = vq_out.permute(0, 2, 1).contiguous()
            if self.project_dim:
                quantized = self.proj_out(vq_out)
            else:
                quantized = vq_out
            return quantized, vq_in, vq_out.detach()

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
                encodings.t(), flatten
            )  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            )
            self.embeddings.data = normalised_updated_ema_w

        # Straight Through Estimator
        vq_out = vq_in + (vq_out - vq_in).detach()

        vq_in = vq_in.permute(0, 2, 1).contiguous()
        vq_out = vq_out.permute(0, 2, 1).contiguous()

        if self.project_dim:
            quantized = self.proj_out(vq_out)
        else:
            quantized = vq_out
        return quantized, vq_in, vq_out.detach()

    def get_code_indices(self, flatten):
        # compute L2 distance
        distances = (
            torch.sum(flatten**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2.0 * torch.matmul(flatten, self.embeddings.t())
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

    def __init__(self, embedding_dim, project_dim, num_quantizers, vq_byte, decay):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_embeddings = 2**vq_byte
        self.num_quantizers = num_quantizers

        self.codebooks = nn.ModuleList(
            [
                VectorQuantize(
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

        vq_in_list = []
        vq_out_detach_list = []

        n_q = self.num_quantizers
        # no dropout

        for layer in self.codebooks[:n_q]:
            quantized, vq_in, vq_out_detach = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            vq_in_list.append(vq_in)
            vq_out_detach_list.append(vq_out_detach)

        return quantized_out, vq_in_list, vq_out_detach_list


class GroupVectorQuantizer(nn.Module):
    """Group VectorQuantizer by using VectorQuantizerEMA1D.
    Flatten the features of each frame, then group these
    features and perform vector quantization.
    """

    def __init__(self, embedding_dim, project_dim, num_groups, vq_byte, decay):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_groups = num_groups
        assert embedding_dim % num_groups == 0, (
            f"Can't group by the setting: embedding_dim={embedding_dim}, "
            f"num_groups={num_groups} "
        )

        self.sub_dim = embedding_dim // num_groups
        self.num_embeddings = 2**vq_byte
        # for each codebook

        self.codebooks = nn.ModuleList(
            [
                VectorQuantize(
                    self.sub_dim, self.project_dim, self.num_embeddings, decay=decay
                )
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

        vq_in_list = []
        vq_out_detach_list = []
        s_q_list = []
        for idx, s in enumerate(torch.chunk(x, self.num_groups, dim=1)):
            # [B, sub_dim, T]
            s_q, vq_in, vq_out_detach = self.codebooks[idx](s)
            vq_in_list.append(vq_in)
            vq_out_detach_list.append(vq_out_detach)
            s_q_list.append(s_q)

        x_q = torch.cat(s_q_list, dim=1)

        return x_q, vq_in_list, vq_out_detach_list

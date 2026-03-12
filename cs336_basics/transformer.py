
import torch.nn as nn
import torch
import math

class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on 
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=dtype)) # (row size, column size)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=2.0 / (out_features + in_features), 
                              a = -3 * math.sqrt(2), b = 3 * math.sqrt(2))

    def forward(self, x):
        return x @ self.weight.T
    
# an embedding layer that maps integer token IDs into a vector space of dimension d_model
# it is just a look up, num_embeddings is all the possible tokenIDs, and embedding_dim is just what this tokenID is
# mapped to 
class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        # device: torch.device | None = None Device to store the parameters on 
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, dtype=dtype)) # (row size, column size)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

# root mean square of norm value
class RMSNormModule(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # d_model: int Hidden dimension of the model
        # eps: float = 1e-5 Epsilon value for numerical stability
        # device: torch.device | None = None Device to store the parameters on 
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype)) # learnable gain parameter
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(x.pow(2).sum(dim=-1, keepdim=True) / x.shape[-1] + self.eps)
        result = x/rms * self.g
        return result.to(in_dtype)

# positionwise_feedforward
class FFNModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1_weight = nn.Parameter(torch.ones(d_ff, d_model, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.ones(d_model, d_ff, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.ones(d_ff, d_model, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(x @ self.w1_weight.T) * (x @ self.w1_weight.T)
        b = x @ self.w3_weight.T
        h = a * b
        result = h @ self.w2_weight.T
        return result


# Relative Positional Embeddings 
# Construct the RoPE module and create buffers if needed.
class RoPEModule(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) 
        # theta: float theta value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(x @ self.w1_weight.T) * (x @ self.w1_weight.T)
        b = x @ self.w3_weight.T
        h = a * b
        result = h @ self.w2_weight.T
        return result

def main():
    """Main entry point of the program."""
    print("NI HOWDY!")

if __name__ == "__main__":
    main()
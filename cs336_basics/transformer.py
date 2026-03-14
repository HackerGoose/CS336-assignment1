
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


# Relative (Rotaty) Positional Embeddings 
# Construct the RoPE module and create buffers if needed.
class RoPEModule(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # theta: float theta value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on
        cos_i_k_map = torch.zeros(max_seq_len, int(d_k/2))
        sin_i_k_map = torch.zeros(max_seq_len, int(d_k/2))
        for i in range(max_seq_len):
            for k in range(int(d_k/2)):
                t = i / (theta ** ((2*k)/d_k))
                cos_i_k_map[i][k] = math.cos(t)
                sin_i_k_map[i][k] = math.sin(t)


        # sin_table : (max_seq_len, d_k/2)
        # cos_table : (max_seq_len, d_k/2)
        # every line represetns position p's corresponded sin/con(theta_p,k)
        # where p is token's position in a seq input and k is vector's 2-elem
        # pair position
        self.register_buffer('cos_i_k', cos_i_k_map, persistent=False)
        self.register_buffer('sin_i_k', sin_i_k_map, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x has shape of (..., seq_length, d_k), output should be the same shape
        # token_positions shape is (..., seq_length), it specifying the token 
        # positions of x along the sequence dimension.
        
        # even/odd slides of x, along vector of size d_k
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2] 

        # use pos to do index on sin_i_k and cos_i_k
        sin = self.sin_i_k[token_positions] 
        cos = self.cos_i_k[token_positions]

        # now shape of sin/cos is (seq_len, d_k/2), and
        # x_even's shape is (..., seq_len, d_k/2), so we need to broadcost
        # cos = torch.broadcast_to(cos, x_even.shape)
        # sin = torch.broadcast_to(sin, x_odd.shape)
        # it is auto broadcosted

        # calculate and do the roation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos 

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)
        return x_rot

# Stable Softmax
def softmax(x, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    # why this is ok? because softmax operation is invariant to adding
    # any constant c to all inputs
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


# Scaled Dot-Product Attention
# Construct the RoPE module and create buffers if needed.
def scaled_dot_product_attention(Q, K, V, mask):
    # Args:
    #     Q (Float[Tensor, " ... queries d_k"]): Query tensor
    #     K (Float[Tensor, " ... keys d_k"]): Key tensor
    #     V (Float[Tensor, " ... values d_v"]): Values tensor
    #     mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    # Returns:
    #     Float[Tensor, " ... queries d_v"]: Output of SDPA
    d_k = Q.shape[-1]
    pre_softmax = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(d_k)
    pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    return softmax(pre_softmax, dim=-1) @ V


def main():
    """Main entry point of the program."""
    print("NI HOWDY!")

if __name__ == "__main__":
    main()
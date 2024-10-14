import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.fasterkan import FasterKAN  # Advanced rational transformer-based architecture
from timm.layers import use_fused_attn

class KAN4MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMREC, self).__init__(config, dataset)

        # Load configuration and dataset parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        # User, item (ID-based), image, and text embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # If available, use pretrained image and text embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_size)
            nn.init.xavier_normal_(self.image_trs.weight)
        else:
            self.image_embedding = nn.Embedding(self.n_items, self.embedding_size)
            nn.init.xavier_uniform_(self.image_embedding.weight)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_size)
            nn.init.xavier_normal_(self.text_trs.weight)
        else:
            self.text_embedding = nn.Embedding(self.n_items, self.embedding_size)
            nn.init.xavier_uniform_(self.text_embedding.weight)

        # KANTransformer for image and text features
        self.kan_user = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout) # For users 
        self.kan_image = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)  # For image interactions
        self.kan_text = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)   # For text interactions

        self.faster_kan = FasterKAN(layers_hidden=[self.embedding_size, self.embedding_size])

    def forward(self):
        # Transform embeddings
        image_embedding_transformed = self.image_trs(self.image_embedding.weight)
        text_embedding_transformed = self.text_trs(self.text_embedding.weight)
        
        # Pass through the rational KAN-based transformer layers
        u_transformed = self.kan_user(self.user_embedding.weight) # [num_users, emb_size]
        i_transformed = self.kan_image(image_embedding_transformed)  # [num_users, num_items]
        t_transformed = self.kan_text(text_embedding_transformed)  # [num_users, num_items]
                
        # Combine user embedding with image and text embeddings
        u_i = torch.matmul(u_transformed, i_transformed.T)  # Shape: [num_users, num_items]
        u_t = torch.matmul(u_transformed, t_transformed.T)  # Shape: [num_users, num_items]

        # FasterKan for transformation u_i, u_t
        u_i = self.faster_kan(u_i)
        u_t = self.faster_kan(u_t)

        return u_i, u_t

    def calculate_loss(self, interaction):
        """
        Calculate the loss using SIGLIP-like approach for user-item interactions, including interaction labels for positive samples.

        Args:
            interaction: Tuple containing users and items (ground-truth interaction matrix [num_users, num_items]),
                         where users have interacted with the corresponding items.
        Returns:
            Total loss for training.
        """
        # Predict interaction scores for u_i and u_t
        u_i, u_t = self.forward()

        # Normalized features
        u_i_transformed = u_i / u_i.norm(p=2, dim=-1, keepdim=True)
        u_t_transformed = u_t / u_t.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity as logits
        logits_u_i = torch.matmul(u_i_transformed, u_i_transformed.t())  # Shape: [num_users, num_users]
        logits_u_t = torch.matmul(u_t_transformed, u_t_transformed.t())  # Shape: [num_users, num_users]

        # Labels for SIGLIP-like contrastive loss
        eye = torch.eye(logits_u_i.size(0), device=logits_u_i.device)
        m1_diag1 = -torch.ones_like(logits_u_i) + 2 * eye
        loglik_u_i = torch.nn.functional.logsigmoid(m1_diag1 * logits_u_i)
        loglik_u_t = torch.nn.functional.logsigmoid(m1_diag1 * logits_u_t)

        # Negative Log Likelihood (NLL) loss for both transformed matrices
        nll_u_i = -torch.sum(loglik_u_i, dim=-1).mean()
        nll_u_t = -torch.sum(loglik_u_t, dim=-1).mean()

        # Interaction-based loss component
        users = interaction[0]  # Batch of users
        items = interaction[1]  # Corresponding items that users interacted with (positive items)

        # Get the interaction scores for these user-item pairs from both image and text transformations
        interaction_u_i_scores = u_i_transformed[users, items]  # Shape: [batch_size]
        interaction_u_t_scores = u_t_transformed[users, items]  # Shape: [batch_size]

        # Labels for these interactions are all 1 (since they are positive samples)
        labels = torch.ones_like(interaction_u_i_scores, device=u_i_transformed.device)

        # Binary Cross-Entropy loss for interaction predictions
        interaction_loss_u_i = torch.nn.functional.binary_cross_entropy_with_logits(interaction_u_i_scores, labels)
        interaction_loss_u_t = torch.nn.functional.binary_cross_entropy_with_logits(interaction_u_t_scores, labels)

        # Regularization term
        reg_term = self.reg_weight * (self.user_embedding.weight.norm() + self.image_embedding.weight.norm() + self.text_embedding.weight.norm())

        # Average loss for transformed u_i and u_t, including interaction loss
        loss = (nll_u_i + nll_u_t).mean() + (interaction_loss_u_i + interaction_loss_u_t).mean() + reg_term + self.cl_weight
        return loss

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for a given user by averaging image and text matrices.

        Args:
            interaction: User interaction data.

        Returns:
            score_mat_ui: Predicted scores for all items.
        """
        user = interaction[0]
        u_i, u_t = self.forward()

        # Get the scores for the given user
        user_image_scores = u_i[user]  # Shape: [num_items]
        user_text_scores = u_t[user]  # Shape: [num_items]

        # Average the scores from image and text models
        score_mat_ui = (user_image_scores + user_text_scores)/2  # Shape: [num_items]

        return score_mat_ui

class KANTransformer(nn.Module):
    """
    KANsiglip class that functions as a transformer-like module with KAN rational activation,
    similar to the image encoder in the SIGLIP model, but enhanced with advanced rational units.
    """
    def __init__(self, embedding_size, n_layers, dropout=0.5):
        super(KANTransformer, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # Transformer-like attention mechanism combined with rational KAN
        self.layers = nn.ModuleList([
            KANLayer(embedding_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        Forward pass for KANsiglip.

        Args:
            x: Input tensor of shape [num_users, num_items].

        Returns:
            Transformed tensor of shape [num_users, num_items].
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
class KANAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class KANLayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class KANLayer(nn.Module):
    def __init__(self, embedding_size, dropout=0.2):
        super(KANLayer, self).__init__()
        # Use the advanced Attention from katransformer.py
        self.attention = KANAttention(dim=embedding_size, num_heads=8, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Use the FasterKAN rational MLP
        self.mlp = FasterKAN(layers_hidden=[embedding_size, embedding_size])

        # Stochastic depth and dropout
        self.drop_path = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.layer_scale = KANLayerScale(dim=embedding_size, init_values=1e-5)  # Layer scaling to stabilize training

    def forward(self, x):
        # Apply attention
        x_attn = self.attention(self.norm1(x))
        x = x + self.drop_path(self.layer_scale(x_attn))

        # Apply rational MLP
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.layer_scale(x_mlp))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.fasterkan import FasterKAN
from timm.layers import use_fused_attn

class KAN4MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMREC, self).__init__(config, dataset)

        # Load configuration and dataset parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        # User, item (ID-based), image, and text embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)

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

        # KANTransformer for image and text features with multi-head attention
        self.kan_user = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)
        self.kan_image = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)
        self.kan_text = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)

        # Use batch normalization and dropout in SplineLinear
        self.SplineLinear = SplineLinear(self.embedding_size, self.embedding_size)
        self.batch_norm = nn.BatchNorm1d(self.n_items)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self):
        # Transform embeddings
        image_embedding_transformed = self.image_trs(self.image_embedding.weight)
        text_embedding_transformed = self.text_trs(self.text_embedding.weight)

        # Pass through the KAN-based transformer layers
        u_transformed = self.kan_user(self.user_embedding.weight)
        i_transformed = self.kan_image(image_embedding_transformed)
        t_transformed = self.kan_text(text_embedding_transformed)

        # Combine user embedding with image and text embeddings
        u_i = torch.matmul(u_transformed, i_transformed.transpose(0, 1))
        u_i = self.SplineLinear(self.batch_norm(u_i))  # Adding batch normalization
        u_i = self.dropout_layer(u_i)

        u_t = torch.matmul(u_transformed, t_transformed.transpose(0, 1))
        u_t = self.SplineLinear(self.batch_norm(u_t))
        u_t = self.dropout_layer(u_t)

        return u_i, u_t

    def calculate_loss(self, interaction):
        u_i, u_t = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        interaction_u_i_scores_pos = u_i[users, pos_items]
        interaction_u_t_scores_pos = u_t[users, pos_items]
        interaction_u_i_scores_neg = u_i[users, neg_items]
        interaction_u_t_scores_neg = u_t[users, neg_items]

        bpr_loss_u_i = -torch.mean(torch.log(torch.sigmoid(interaction_u_i_scores_pos - interaction_u_i_scores_neg) + 1e-10))
        bpr_loss_u_t = -torch.mean(torch.log(torch.sigmoid(interaction_u_t_scores_pos - interaction_u_t_scores_neg) + 1e-10))

        # Include L2 regularization to avoid overfitting
        l2_reg = 1e-5 * (torch.sum(self.user_embedding.weight ** 2) + torch.sum(self.image_embedding.weight ** 2) + torch.sum(self.text_embedding.weight ** 2))

        loss = (bpr_loss_u_i + bpr_loss_u_t) / 2 + self.cl_weight + l2_reg
        print(f"Total loss: {loss}")
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        u_i, u_t = self.forward()

        user_image_scores = u_i[users]
        user_text_scores = u_t[users]

        # Concatenate the scores and transform via linear layer for better alignment
        combined_scores = torch.cat([user_image_scores, user_text_scores], dim=1)
        score_mat_ui = self.SplineLinear(combined_scores)

        return score_mat_ui
    
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization

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

    def forward(self, x: torch.Tensor):
        """
        Forward pass for KANsiglip.

        Args:
            x: Input tensor of shape [seq_len, embedding_size].

        Returns:
            Transformed tensor of shape [seq_len, num_items].
        """
        for layer in self.layers:
            x = layer(x)
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
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.hidden_size = 64

        # Use the FasterKAN rational MLP
        self.FasterKAN = FasterKAN(layers_hidden=[embedding_size, self.hidden_size, embedding_size])

        # Stochastic depth and dropout
        self.drop_path = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.layer_scale = KANLayerScale(dim=embedding_size, init_values=1e-5)  # Layer scaling to stabilize training

    def forward(self, hidden_states: torch.Tensor):
        # Apply attention
        residual = hidden_states
        hidden_states = self.drop_path(self.layer_scale(self.norm1(hidden_states)))

        hidden_states = hidden_states + residual
        residual = hidden_states
        # Apply FasterKAN
        hidden_states = self.FasterKAN(self.norm2(hidden_states))
        hidden_states = residual + self.drop_path(self.layer_scale(hidden_states))
        return hidden_states

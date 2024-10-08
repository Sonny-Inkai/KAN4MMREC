import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.fasterkan import FasterKAN  # Advanced rational transformer-based architecture
from utils.katransformer import Attention, LayerScale

class KAN4MMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMRec, self).__init__(config, dataset)

        # Load configuration and dataset parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        self.n_users = self.n_users  # From GeneralRecommender
        self.n_items = self.n_items  # From GeneralRecommender

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

        # KANsiglip Transformers for image and text features
        self.kan_siglip_image = KANsiglip(self.embedding_size, self.n_layers, dropout=self.dropout)  # For user-image interactions
        self.kan_siglip_text = KANsiglip(self.embedding_size, self.n_layers, dropout=self.dropout)   # For user-text interactions

    def forward(self):
        # Combine user embedding with image and text embeddings
        u_i = torch.matmul(self.user_embedding.weight, self.image_embedding.weight.T)  # Shape: [num_users, num_items]
        u_t = torch.matmul(self.user_embedding.weight, self.text_embedding.weight.T)  # Shape: [num_users, num_items]

        # Pass through the rational KAN-based transformer layers
        u_i_transformed = self.kan_siglip_image(u_i)  # [num_users, num_items]
        u_t_transformed = self.kan_siglip_text(u_t)  # [num_users, num_items]

        return u_i_transformed, u_t_transformed

    def calculate_loss(self, interaction_matrix):
        """
        Calculate the loss using SIGLIP-like approach for user-item interactions.

        Args:
            interaction_matrix: Ground-truth interaction matrix [num_users, num_items], where 1 means interaction and 0 means no interaction.
        Returns:
            Total loss for training.
        """
        # Predict interaction scores for u_i and u_t
        u_i_transformed, u_t_transformed = self.forward()

        # Normalized features
        u_i_transformed = u_i_transformed / u_i_transformed.norm(p=2, dim=-1, keepdim=True)
        u_t_transformed = u_t_transformed / u_t_transformed.norm(p=2, dim=-1, keepdim=True)

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

        # Average loss for transformed u_i and u_t
        loss = (nll_u_i + nll_u_t) / 2 + self.reg_weight + self.cl_weight
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
        user_image_transformed, user_text_transformed = self.forward()

        # Get the scores for the given user
        user_image_scores = user_image_transformed[user]  # Shape: [num_items]
        user_text_scores = user_text_transformed[user]  # Shape: [num_items]

        # Average the scores from image and text models
        score_mat_ui = (user_image_scores + user_text_scores) / 2  # Shape: [num_items]

        return score_mat_ui

class KANsiglip(nn.Module):
    """
    KANsiglip class that functions as a transformer-like module with KAN rational activation,
    similar to the image encoder in the SIGLIP model, but enhanced with advanced rational units.
    """
    def __init__(self, embedding_size, n_layers, dropout=0.5):
        super(KANsiglip, self).__init__()
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


class KANLayer(nn.Module):
    def __init__(self, embedding_size, dropout=0.2):
        super(KANLayer, self).__init__()
        # Use the advanced Attention from katransformer.py
        self.attention = Attention(dim=embedding_size, num_heads=8, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Use the FasterKAN rational MLP
        self.mlp = FasterKAN(layers_hidden=[embedding_size, embedding_size])

        # Stochastic depth and dropout
        self.drop_path = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.layer_scale = LayerScale(dim=embedding_size, init_values=1e-5)  # Layer scaling to stabilize training

    def forward(self, x):
        # Apply attention
        x_attn = self.attention(self.norm1(x))
        x = x + self.drop_path(self.layer_scale(x_attn))

        # Apply rational MLP
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.layer_scale(x_mlp))
        return x


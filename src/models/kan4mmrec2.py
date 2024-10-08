import torch
import torch.nn as nn
import torch.nn.functional as F
from kat_rational import KAT_Group  # Custom rational activation function group
from common.abstract_recommender import GeneralRecommender
from fasterkan import FasterKAN  # Advanced rational transformer-based architecture
from katransformer import Attention, LayerScale

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
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

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
        self.kan_siglip_image = KANSIGLIP(self.embedding_size, self.n_layers, dropout=self.dropout)  # For user-image interactions
        self.kan_siglip_text = KANSIGLIP(self.embedding_size, self.n_layers, dropout=self.dropout)   # For user-text interactions

    def forward(self):
        # Combine user embedding with image and text embeddings
        user_image_interaction = torch.matmul(self.user_embedding.weight, self.image_embedding.weight.T)  # Shape: [num_users, num_items]
        user_text_interaction = torch.matmul(self.user_embedding.weight, self.text_embedding.weight.T)  # Shape: [num_users, num_items]

        # Pass through the rational KAN-based transformer layers
        user_image_transformed = self.kan_siglip_image(user_image_interaction)  # Shape: [num_users, num_items]
        user_text_transformed = self.kan_siglip_text(user_text_interaction)  # Shape: [num_users, num_items]

        return user_image_transformed, user_text_transformed

    def calculate_loss(self, interaction_matrix):
        """
        Calculate the loss using SIGLIP-like approach for user-item interactions.

        Args:
            interaction_matrix: Ground-truth interaction matrix [num_users, num_items], where 1 means interaction and 0 means no interaction.
        Returns:
            Total loss for training.
        """
        # Predict interaction scores for user_image and user_text
        user_image_transformed, user_text_transformed = self.forward()

        # Normalized features
        user_image_transformed = user_image_transformed / user_image_transformed.norm(p=2, dim=-1, keepdim=True)
        user_text_transformed = user_text_transformed / user_text_transformed.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity as logits
        logits_user_image = torch.matmul(user_image_transformed, user_image_transformed.t())  # Shape: [num_users, num_users]
        logits_user_text = torch.matmul(user_text_transformed, user_text_transformed.t())  # Shape: [num_users, num_users]

        # Labels for SIGLIP-like contrastive loss
        eye = torch.eye(logits_user_image.size(0), device=logits_user_image.device)
        m1_diag1 = -torch.ones_like(logits_user_image) + 2 * eye
        loglik_user_image = torch.nn.functional.logsigmoid(m1_diag1 * logits_user_image)
        loglik_user_text = torch.nn.functional.logsigmoid(m1_diag1 * logits_user_text)

        # Negative Log Likelihood (NLL) loss for both transformed matrices
        nll_user_image = -torch.sum(loglik_user_image, dim=-1).mean()
        nll_user_text = -torch.sum(loglik_user_text, dim=-1).mean()

        # Average loss for transformed user_image and user_text
        loss = (nll_user_image + nll_user_text) / 2
        return loss
    
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_image_transformed, user_text_transformed = self.forward()


class KANSIGLIP(nn.Module):
    """
    KANsiglip class that functions as a transformer-like module with KAN rational activation,
    similar to the image encoder in the SIGLIP model, but enhanced with advanced rational units.
    """
    def __init__(self, embedding_size, n_layers, dropout=0.5):
        super(KANSIGLIP, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # Transformer-like attention mechanism combined with rational KAN
        self.layers = nn.ModuleList([
            KANLayer(embedding_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, interaction_matrix):
        """
        Forward pass for KANsiglip.

        Args:
            interaction_matrix: Input tensor of shape [num_users, num_items].

        Returns:
            Transformed tensor of shape [num_users, num_items].
        """
        for layer in self.layers:
            interaction_matrix = layer(interaction_matrix)
        return interaction_matrix


class KANLayer(nn.Module):
    def __init__(self, embedding_size, dropout=0.5):
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

    def forward(self, interaction_matrix):
        # Apply attention
        normalized_interaction_matrix = self.norm1(interaction_matrix)  # [Batch_Size, Num_Items]
        attention_output = self.attention(normalized_interaction_matrix)  # [Batch_Size, Num_Items]
        interaction_matrix = interaction_matrix + self.drop_path(self.layer_scale(attention_output))

        # Apply rational MLP
        normalized_interaction_matrix = self.norm2(interaction_matrix)  # [Batch_Size, Num_Items]
        mlp_output = self.mlp(normalized_interaction_matrix)  # [Batch_Size, Num_Items]
        interaction_matrix = interaction_matrix + self.drop_path(self.layer_scale(mlp_output))
        return interaction_matrix
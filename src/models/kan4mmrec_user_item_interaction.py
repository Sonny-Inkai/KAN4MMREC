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

        self.FasterKAN = FasterKAN(layers_hidden=[self.t_feat.shape[0], self.t_feat.shape[0]])

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
        u_i = self.FasterKAN(u_i)

        u_t = torch.matmul(u_transformed, t_transformed.T)  # Shape: [num_users, num_items]
        u_t = self.FasterKAN(u_t)

        return u_i, u_t

    def calculate_loss(self, interaction):
        """
        Calculate the loss using SIGLIP-like approach for user-user , and loss including interaction labels for positive samples.

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

        # Interaction-based loss component
        users = interaction[0]  # Corresponding items that users interacted with (positive items)
        pos_items = interaction[1] # Positive items
        neg_items = interaction[2]  # Negative sampled items

        # Get the interaction scores for these user-item pairs from both image and text transformations
        interaction_u_i_scores_pos = u_i_transformed[users, pos_items]  # Shape: [batch_size]

        interaction_u_t_scores_pos = u_t_transformed[users, pos_items]  # Shape: [batch_size]

        interaction_u_i_scores_neg = u_i_transformed[users, neg_items]  # Shape: [batch_size, num_neg_samples]

        interaction_u_t_scores_neg = u_t_transformed[users, neg_items]  # Shape: [batch_size, num_neg_samples]

        # BPR Loss for interaction predictions
        bpr_loss_u_i = -torch.mean(torch.log2(torch.sigmoid(interaction_u_i_scores_pos - interaction_u_i_scores_neg).sum(dim=-1)))
        bpr_loss_u_t = -torch.mean(torch.log2(torch.sigmoid(interaction_u_t_scores_pos - interaction_u_t_scores_neg).sum(dim=-1)))

        bpr_loss_i_t = -torch.mean(torch.log2(torch.sigmoid(interaction_u_i_scores_pos-interaction_u_t_scores_neg).sum(dim=-1)))
        bpr_loss_t_i = -torch.mean(torch.log2(torch.sigmoid(interaction_u_t_scores_pos-interaction_u_i_scores_neg).sum(dim=-1)))

        print("BPR Loss for u_i:", bpr_loss_u_i.item())
        print("BPR Loss for u_t:", bpr_loss_u_t.item())
        print("BPR loss for image and text:", bpr_loss_i_t)
        print("BPR loss for text and image:", bpr_loss_t_i)
        # Average loss for transformed u_i and u_t, including interaction loss
        loss = (bpr_loss_u_i + bpr_loss_u_t).mean() + (bpr_loss_i_t + bpr_loss_t_i).mean() + self.cl_weight
        print("Total Loss:", loss.item())
        return loss

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for a given user by averaging image and text matrices.

        Args:
            interaction: User interaction data.

        Returns:
            score_mat_ui: Predicted scores for all items.
        """
        users = interaction[0]
        u_i, u_t = self.forward()

        # Get the scores for the given user
        user_image_scores = u_i[users]  # Shape: [num_items]
        user_text_scores = u_t[users]  # Shape: [num_items]

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

        # Use the FasterKAN rational MLP
        self.FasterKAN = FasterKAN(layers_hidden=[embedding_size, embedding_size])

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

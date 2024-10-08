import torch
import torch.nn as nn
import torch.nn.functional as F
from kat_rational import KAT_Group  # Custom rational activation function group
from common.abstract_recommender import GeneralRecommender
from fasterkan import FasterKAN  # Advanced rational transformer-based architecture
from katransformer import Attention, LayerScale
from typing import Optional, Tuple


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
        loss = (nll_u_i + nll_u_t) / 2
        return loss




class KANSIGLIPConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        use_rational_mlp=True,
        num_image_tokens: int = None,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        self.use_rational_mlp = use_rational_mlp


class KANSIGLIPEmbeddings(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class KANSIGLIPAttention(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class KANSIGLIPMLP(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = FasterKAN(layers_hidden=[config.hidden_size, config.intermediate_size, config.hidden_size])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class KANSIGLIPEncoderLayer(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.self_attn = KANSIGLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = KANSIGLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class KANSIGLIPEncoder(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [KANSIGLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class KANSIGLIPTransformer(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.embeddings = KANSIGLIPEmbeddings(config)
        self.encoder = KANSIGLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class KANSIGLIPModel(nn.Module):
    def __init__(self, config: KANSIGLIPConfig):
        super().__init__()
        self.vision_model = KANSIGLIPTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)

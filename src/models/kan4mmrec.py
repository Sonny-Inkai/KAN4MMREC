import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class KAN4MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMREC, self).__init__(config, dataset)

        # Load configuration and dataset parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout']

        # User, item (ID-based), image, and text embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # Pre-trained image and text embeddings
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

        self.layer_norm = nn.LayerNorm(self.embedding_size)

        # TransformerEncoder for user, image, and text
        encoder_layers = TransformerEncoderLayer(d_model=self.embedding_size, nhead=8, dropout=self.dropout)
        self.user_encoder = TransformerEncoder(encoder_layers, num_layers=self.n_layers)
        self.image_encoder = TransformerEncoder(encoder_layers, num_layers=self.n_layers)
        self.text_encoder = TransformerEncoder(encoder_layers, num_layers=self.n_layers)

        # Linear layers for final prediction
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=8)

    def forward(self):
        # Transform embeddings
        image_embedding_transformed = self.image_trs(self.image_embedding.weight)
        text_embedding_transformed = self.text_trs(self.text_embedding.weight)

        # Pass through transformer encoder layers
        u_transformed = self.user_encoder(self.user_embedding.weight)
        i_transformed = self.image_encoder(image_embedding_transformed)
        t_transformed = self.text_encoder(text_embedding_transformed)

        return u_transformed, i_transformed, t_transformed

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        # Predict interaction scores for u_i and u_t
        u_transformed, i_transformed, t_transformed = self.forward()

        # Interaction-based loss component
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Cross-attention to combine image and text features
        attn_output, _ = self.cross_attention(u_transformed[users].unsqueeze(0), 
                                              i_transformed[pos_items].unsqueeze(0),
                                              t_transformed[pos_items].unsqueeze(0))
        attn_output = attn_output.squeeze(0)

        mf_v_loss = self.bpr_loss(u_transformed[users], i_transformed[pos_items], i_transformed[neg_items])
        mf_t_loss = self.bpr_loss(u_transformed[users], t_transformed[pos_items], t_transformed[neg_items])

        total_loss = self.cl_weight * torch.norm(attn_output) + mf_t_loss + mf_v_loss
        print(f"Total Loss: {total_loss}")
        return total_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        u_transformed, i_transformed, t_transformed = self.forward()

        # Get the scores for the given user
        u_i = torch.matmul(u_transformed[users], i_transformed.transpose(0, 1))
        u_t = torch.matmul(u_transformed[users], t_transformed.transpose(0, 1))

        # Use cross-attention to adjust scores
        u_combined = (u_i + u_t) / 2

        return u_combined


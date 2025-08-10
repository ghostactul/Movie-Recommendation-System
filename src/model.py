# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class NCF(nn.Module):
    # def __init__(self, num_users, num_items, embedding_dim=128,
    #              mlp_layers=(128, 64, 32, 16), dropout=0.2, normalize_embeddings=False):
    #     super().__init__()
    #     self.num_users = num_users
    #     self.num_items = num_items
    #     self.normalize_embeddings = normalize_embeddings
        
    #     # GMF embeddings
    #     self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
    #     self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
    #     # MLP embeddings
    #     self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
    #     self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
    #     # MLP layers
    #     mlp_modules = []
    #     input_size = embedding_dim * 2
    #     for layer_size in mlp_layers:
    #         mlp_modules.append(nn.Linear(input_size, layer_size))
    #         mlp_modules.append(nn.ReLU())
    #         mlp_modules.append(nn.Dropout(dropout))
    #         input_size = layer_size
    #     self.mlp_layers = nn.Sequential(*mlp_modules)
        
    #     # Final prediction layer
    #     self.predict_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
    #     self._init_weights()

    # def _init_weights(self):
    #     # Embedding initialization
    #     nn.init.xavier_uniform_(self.user_embedding_gmf.weight)
    #     nn.init.xavier_uniform_(self.item_embedding_gmf.weight)
    #     nn.init.xavier_uniform_(self.user_embedding_mlp.weight)
    #     nn.init.xavier_uniform_(self.item_embedding_mlp.weight)

    #     # Linear layers
    #     for layer in self.mlp_layers:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_uniform_(layer.weight)
    #             nn.init.zeros_(layer.bias)
        
    #     nn.init.xavier_uniform_(self.predict_layer.weight)
    #     nn.init.zeros_(self.predict_layer.bias)

    # def forward(self, user_indices, item_indices):
    #     # GMF branch
    #     user_embed_gmf = self.user_embedding_gmf(user_indices)
    #     item_embed_gmf = self.item_embedding_gmf(item_indices)
    #     gmf_output = user_embed_gmf * item_embed_gmf  # element-wise
        
    #     # MLP branch
    #     user_embed_mlp = self.user_embedding_mlp(user_indices)
    #     item_embed_mlp = self.item_embedding_mlp(item_indices)
    #     mlp_input = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
    #     mlp_output = self.mlp_layers(mlp_input)
        
    #     # L2 normalization for stability in BPR
    #     if self.normalize_embeddings:
    #         gmf_output = F.normalize(gmf_output, p=2, dim=-1)
    #         mlp_output = F.normalize(mlp_output, p=2, dim=-1)
        
    #     # Concatenate GMF and MLP outputs
    #     concat = torch.cat([gmf_output, mlp_output], dim=-1)
        
    #     prediction = self.predict_layer(concat).squeeze(-1)  # raw score for BPR
    #     return prediction
import torch
import torch.nn as nn

class NCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        # GMF embeddings
        self.user_embed_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embed_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.user_embed_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embed_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(embedding_dim + 32, 1)

    def forward(self, user, item):
        gmf_out = self.user_embed_gmf(user) * self.item_embed_gmf(item)
        mlp_input = torch.cat([self.user_embed_mlp(user), self.item_embed_mlp(item)], dim=-1)
        mlp_out = self.mlp(mlp_input)
        final_input = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output_layer(final_input).squeeze()


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

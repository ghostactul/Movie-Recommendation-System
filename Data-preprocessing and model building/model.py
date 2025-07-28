import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        
        # GMF part
        self.user_embed_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embed_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP part
        self.user_embed_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embed_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Final layer
        self.output_layer = nn.Linear(embedding_dim + 1, 1)  # 1 for the MLP output (single prediction)

    def forward(self, user, item):
        # GMF part
        gmf_user = self.user_embed_gmf(user)
        gmf_item = self.item_embed_gmf(item)
        gmf_out = gmf_user * gmf_item  # Element-wise multiplication for GMF

        # MLP part
        mlp_user = self.user_embed_mlp(user)
        mlp_item = self.item_embed_mlp(item)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)  # MLP outputs a single value for each user-item pair
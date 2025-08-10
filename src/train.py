# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import os

# from torch.optim import AdamW
# from torch.optim.lr_scheduler import StepLR

# def bpr_loss(pos_scores, neg_scores):
#     loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
#     return loss

# def train_ncf_bpr(model, train_dataset, device='cpu',
#                   epochs=30, batch_size=512, lr=5e-4,
#                   save_path=None):
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(1, epochs+1):
#         model.train()
#         total_loss = 0.0

#         for users, items in train_loader:
#             users = users.to(device)
#             items = items.to(device)

#             pos_items = items[:, 0]
#             neg_items = items[:, 1:].reshape(-1)  # flatten negatives
#             users_expanded = users[:, 0].repeat_interleave(items.size(1) - 1)

#             optimizer.zero_grad()
#             pos_scores = model(users[:, 0], pos_items)
#             neg_scores = model(users_expanded, neg_items)

#             loss = bpr_loss(pos_scores.repeat_interleave(items.size(1) - 1), neg_scores)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             total_loss += loss.item() * users.size(0)

#         scheduler.step()
#         avg_loss = total_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch}/{epochs} train_bpr_loss={avg_loss:.4f}")

#         if save_path:
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             torch.save({'model_state_dict': model.state_dict()}, save_path)

#     return model

import torch
from torch.utils.data import DataLoader
from model import bpr_loss

def train_ncf_bpr(model, dataset, device='cpu',
                  epochs=20, batch_size=1024, lr=0.001,
                  weight_decay=1e-6, save_path=None):
    """
    Train NCF using BPR loss with the better-performing settings.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, pos_item, neg_item in train_loader:
            user, pos_item, neg_item = user.to(device), pos_item.to(device), neg_item.to(device)

            pos_scores = model(user, pos_item)
            neg_scores = model(user, neg_item)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, BPR Loss: {avg_loss:.4f}")

    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

    return model

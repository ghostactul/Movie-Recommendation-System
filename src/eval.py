# # src/eval.py
# import numpy as np

# def precision_at_k(recs, truth, k):
#     if not truth: return 0.0
#     recs_k = recs[:k]
#     hits = sum([1 for r in recs_k if r in truth])
#     return hits / k

# def recall_at_k(recs, truth, k):
#     if not truth: return 0.0
#     recs_k = recs[:k]
#     hits = sum([1 for r in recs_k if r in truth])
#     return hits / len(truth)

# def dcg_at_k(recs, truth, k):
#     dcg = 0.0
#     for i, r in enumerate(recs[:k]):
#         rel = 1.0 if r in truth else 0.0
#         if i == 0:
#             dcg += rel
#         else:
#             dcg += rel / np.log2(i+1+0)
#     return dcg

# def ndcg_at_k(recs, truth, k):
#     if not truth: return 0.0
#     idcg = 0.0
#     # ideal has min(len(truth), k) ones at top
#     for i in range(min(len(truth), k)):
#         if i == 0:
#             idcg += 1.0
#         else:
#             idcg += 1.0 / np.log2(i+1+0)
#     dcg = dcg_at_k(recs, truth, k)
#     return dcg / idcg if idcg > 0 else 0.0
import torch
import numpy as np
from tqdm import tqdm

def hit_ratio(ranklist, gt_item):
    """Hit Ratio @K: 1 if ground truth is in top-K, else 0."""
    return 1 if gt_item in ranklist else 0

def ndcg(ranklist, gt_item):
    """NDCG @K: Normalized Discounted Cumulative Gain."""
    if gt_item in ranklist:
        index = ranklist.index(gt_item)
        return np.log(2) / np.log(index + 2)
    return 0

def precision_at_k(ranklist, gt_item, k):
    """Precision@K for a single user (binary ground truth)."""
    return 1.0 / k if gt_item in ranklist else 0.0

def recall_at_k(ranklist, gt_item, total_relevant):
    """Recall@K for a single user (binary ground truth)."""
    return 1.0 if gt_item in ranklist else 0.0

def evaluate_model(model, train_df, test_df, num_items, K=10, device='cpu'):
    """
    Evaluate model on leave-one-out test set.
    Each test instance: one positive (last interaction) + sampled negatives.
    """
    model.eval()
    hits, ndcgs, precisions, recalls = [], [], [], []

    user_item_train = train_df.groupby('user')['movie'].apply(set).to_dict()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        user = int(row['user'])
        gt_item = int(row['movie'])

        # Generate negatives
        negatives = []
        while len(negatives) < 99:  # 99 negatives + 1 positive
            neg = np.random.randint(0, num_items)
            if neg not in user_item_train.get(user, set()) and neg != gt_item:
                negatives.append(neg)

        # Candidate items
        items = negatives + [gt_item]
        users = [user] * len(items)

        # Convert to tensors
        users_tensor = torch.tensor(users, dtype=torch.long, device=device)
        items_tensor = torch.tensor(items, dtype=torch.long, device=device)

        # Predict scores
        with torch.no_grad():
            scores = model(users_tensor, items_tensor).cpu().numpy()

        # Rank top-K
        item_score_dict = {item: score for item, score in zip(items, scores)}
        ranked_items = sorted(item_score_dict, key=item_score_dict.get, reverse=True)[:K]

        # Metrics
        hits.append(hit_ratio(ranked_items, gt_item))
        ndcgs.append(ndcg(ranked_items, gt_item))
        precisions.append(precision_at_k(ranked_items, gt_item, K))
        recalls.append(recall_at_k(ranked_items, gt_item, 1))  # Only 1 positive

    hr = np.mean(hits)
    ndcg_score = np.mean(ndcgs)
    precision_score = np.mean(precisions)
    recall_score = np.mean(recalls)

    return hr, ndcg_score, precision_score, recall_score

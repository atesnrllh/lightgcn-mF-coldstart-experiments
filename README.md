# LightGCN & MF – Cold-Start Embedding Experiments and Dataset Consistency Analysis

This repository contains small, controlled experiments comparing  
**Matrix Factorization (MF)** and **LightGCN** under **cold-start item** conditions.

The goal is not to build full recommendation models, but to clearly observe:

- How item embeddings update when an item appears only as a **positive**,  
- How embeddings update when an item appears only as a **negative**,  
- Whether an isolated item (no interactions at all) is trainable,  
- How LightGCN’s propagation step (`A · E`) affects gradient flow,  
- Why LightGCN behaves differently from MF for cold-start items.

In addition to cold-start embedding experiments, the repository also includes a
small analysis utility for checking train/test user–item consistency and
interaction sparsity in the used LightGCN datasets
(Gowalla, Yelp2018, Amazon-Book).

---

## 📁 Files

### **1. `1_mf_cold_item_embedding_check.py`**
A simple MF implementation used to examine cold-start behavior.

Key points:
- Defines a toy set of user–item interactions.
- Includes items with positive interactions, negative-only items, and unused items.
- Updates come purely from the **BPR loss** (no propagation).
- Items that never appear in the loss remain unchanged.

---

### **2. `2_lightgcn_cold_item_embedding_check.py`**
A LightGCN-style version using the same data setup.

Key points:
- Combines user and item embeddings, then applies:
  \[
  E' = A \cdot E
  \]
- Isolated items receive a **zero propagation vector**.
- Loss cannot update embeddings that do not influence the forward pass.
- Demonstrates why LightGCN struggles with items with no neighbors.

---
### **3. `3_check_lightgcn_train_test_overlap.py`**
A dataset-level analysis script for verifying train/test consistency and
basic sparsity characteristics in the three official LightGCN datasets
(Gowalla, Yelp2018, Amazon-Book).

Key points:

- Checks whether all test users appear in train
- Checks whether all test items appear in train
- Reports total numbers of users, items, and interactions
- Computes train/test interaction ratio
- Identifies items appearing only in train

The datasets follow the interaction format:
```
user item1 item2 item3 ...
```
---

## 🔍 What the Experiments Show

### **MF**
- Any item appearing in the **loss** (positive or negative) receives gradients.
- Completely unused items (never positive, never negative) do **not** update.
- Negative-only items **do** update, because the loss depends on their scores.

### **LightGCN**
- Propagation uses the normalized adjacency matrix.
- An item with **no edges** gets:
  \[
  E'[i] = 0
  \]
- If the model’s score does not depend on a trainable parameter,  
  its gradient remains zero.
- Therefore isolated items remain unchanged, even if sampled as negatives.

### **The Outputs of `3_check_lightgcn_train_test_overlap.py`**
```
===== DATASET: gowalla =====
------------------------------------------------------------
# train users        : 29858
# test users         : 29858
# train items        : 40981
# test items         : 38546
# train interactions : 810128
# test interactions  : 217242
train/test interaction ratio : 3.729
------------------------------------------------------------
Test users ⊆ Train users ?  True
Users in test but NOT in train: 0
Test items ⊆ Train items ?  True
Items in test but NOT in train: 0
------------------------------------------------------------
Users ONLY in train : 0
Users ONLY in test  : 0
Items ONLY in train : 2435
Items ONLY in test  : 0
------------------------------------------------------------

===== DATASET: yelp2018 =====
------------------------------------------------------------
# train users        : 31668
# test users         : 31668
# train items        : 38048
# test items         : 36073
# train interactions : 1237259
# test interactions  : 324147
train/test interaction ratio : 3.817
------------------------------------------------------------
Test users ⊆ Train users ?  True
Users in test but NOT in train: 0
Test items ⊆ Train items ?  True
Items in test but NOT in train: 0
------------------------------------------------------------
Users ONLY in train : 0
Users ONLY in test  : 0
Items ONLY in train : 1975
Items ONLY in test  : 0
------------------------------------------------------------

===== DATASET: amazon-book =====
------------------------------------------------------------
# train users        : 52643
# test users         : 52643
# train items        : 91599
# test items         : 82629
# train interactions : 2380730
# test interactions  : 603378
train/test interaction ratio : 3.946
------------------------------------------------------------
Test users ⊆ Train users ?  True
Users in test but NOT in train: 0
Test items ⊆ Train items ?  True
Items in test but NOT in train: 0
------------------------------------------------------------
Users ONLY in train : 0
Users ONLY in test  : 0
Items ONLY in train : 8970
Items ONLY in test  : 0
------------------------------------------------------------

```
---

## ▶️ Running the Scripts

```bash
python 1_mf_cold_item_embedding_check.py
python 2_lightgcn_cold_item_embedding_check.py
python 3_check_lightgcn_train_test_overlap.py
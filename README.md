# LightGCN & MF – Cold-Start Embedding Experiments

This repository contains small, controlled experiments comparing  
**Matrix Factorization (MF)** and **LightGCN** under **cold-start item** conditions.

The goal is not to build full recommendation models, but to clearly observe:

- How item embeddings update when an item appears only as a **positive**,  
- How embeddings update when an item appears only as a **negative**,  
- Whether an isolated item (no interactions at all) is trainable,  
- How LightGCN’s propagation step (`A · E`) affects gradient flow,  
- Why LightGCN behaves differently from MF for cold-start items.

The scripts are intentionally minimal to make gradient flow and propagation behavior easy to inspect.

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

---

## ▶️ Running the Scripts

```bash
python 1_mf_cold_item_embedding_check.py
python 2_lightgcn_cold_item_embedding_check.py

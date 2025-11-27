"""
mf_cold_item_embedding_check.py

Bu script, saf MF (Matrix Factorization) modelinde hangi item embedding'lerinin
güncellendiğini görmek için yazılmıştır.

Özellikle:
- Item 0–6: Pozitif etkileşimleri var → train'de POSITIF örnek olarak kullanılıyor.
- Item 9 : Hiç pozitif etkileşimi yok, sadece NEGATIF örnek havuzunda.
- Item 7–8: Ne pozitif etkileşimde var ne negatif havuzda → tamamen "cold" ve kullanılmıyor.

Amaç:
- Sadece MF varken (propagation yokken),
  pozitif olan, sadece negatif olan ve hiç kullanılmayan item'ların
  embedding değişimlerini gözlemek.
"""

"""
NOT: ITEM INDEKSLERININ AÇIK HARITASI
------------------------------------

Bu script'te user ve item id'leri şöyle tanımlanmıştır:

  User id'leri : 0 .. NUM_USERS-1
  Item id'leri : 0 .. NUM_ITEMS-1

Örnek (NUM_USERS = 10, NUM_ITEMS = 10 için):

  USER ID'LERI:
      User 0
      User 1
      User 2
      User 3
      User 4
      User 5
      User 6
      User 7
      User 8
      User 9

  ITEM ID'LERI:
      Item 0
      Item 1
      Item 2
      Item 3
      Item 4
      Item 5
      Item 6
      Item 7
      Item 8
      Item 9

Bu MF script'inde graph yoktur, adjacency yoktur.
Doğrudan user_emb ve item_emb üzerinden skor hesaplanır:

  score(u, i) = < user_emb[u], item_emb[i] >

Bu nedenle:
- Pozitif veya negatif örnek olarak loss'a giren her item embedding'i gradient alır.
- Hiç pozitif/negatif seçilmeyen item embedding'leri ilk init değerinde kalır.
------------------------------------
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# ============================================================
# 1. Genel ayarlar
# ============================================================

torch.manual_seed(42)
random.seed(42)

NUM_USERS = 10
NUM_ITEMS = 10
EMB_DIM   = 4


# ============================================================
# 2. Yardımcı: kullanıcı ve item indexlerini göster
# ============================================================

def print_index_info():
    print("==== USER INDEXLERI ====")
    print(list(range(NUM_USERS)))
    print("\n==== ITEM INDEXLERI ====")
    print(list(range(NUM_ITEMS)))


# ============================================================
# 3. Train etkileşimleri ve negatif havuz
# ============================================================

# (user, item) pozitif etkileşimleri
TRAIN_INTERACTIONS = [
    (0, 0), (0, 1),
    (1, 2), (1, 3),
    (2, 4), (3, 5),
    (4, 6), (5, 0),
    (6, 1), (7, 2),
]


def print_interactions_and_pools():
    print("\n==== POZITIF ETKILESIMLER (user, item) ====")
    for u, i in TRAIN_INTERACTIONS:
        print(f"User {u} --> Item {i}")

    items_with_pos = {i for _, i in TRAIN_INTERACTIONS}  # {0,1,2,3,4,5,6}

    # Negatif havuz: pozitif item'lar + sadece-negatif item 9
    negative_pool = sorted(list(items_with_pos | {9}))

    print("\n==== NEGATIF HAVUZUNDAKI ITEM'LAR ====")
    print(negative_pool)

    print("\n==== HIC KULLANILMAYAN ITEM'LAR ====")
    print([7, 8])

    return items_with_pos, negative_pool


# ============================================================
# 4. Model: Simple MF (propagation yok)
# ============================================================

class SimpleMF(nn.Module):
    """
    LightGCN embedding mantığını basitleştiren MF modeli.

    Burada:
      - user_emb: [NUM_USERS, EMB_DIM]
      - item_emb: [NUM_ITEMS, EMB_DIM]
      - skor: dot(user_emb[u], item_emb[i])

    Yani graf yok, propagation yok:
      score(u, i) = < user_emb[u], item_emb[i] >
    """
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def forward(self, u, i):
        u_e = self.user_emb(u)      # [batch, d]
        i_e = self.item_emb(i)      # [batch, d]
        scores = (u_e * i_e).sum(dim=1)  # dot product
        return scores


# ============================================================
# 5. BPR örnekleme ve loss
# ============================================================

def sample_batch(batch_size, train_interactions, negative_pool):
    """
    Basit BPR örnekleri (u, pos, neg) üretir.
    - u, pos: TRAIN_INTERACTIONS içinden
    - neg   : negative_pool içinden, pos'tan farklı olacak şekilde
    """
    batch_u, batch_pos, batch_neg = [], [], []

    for _ in range(batch_size):
        u, pos = random.choice(train_interactions)
        neg = random.choice(negative_pool)
        while neg == pos:
            neg = random.choice(negative_pool)

        batch_u.append(u)
        batch_pos.append(pos)
        batch_neg.append(neg)

    return (
        torch.tensor(batch_u, dtype=torch.long),
        torch.tensor(batch_pos, dtype=torch.long),
        torch.tensor(batch_neg, dtype=torch.long),
    )


def bpr_loss(model, u, pos, neg):
    """
    Basit BPR loss:
      L = - E[ log sigma( s(u,pos) - s(u,neg) ) ]
    """
    pos_scores = model(u, pos)
    neg_scores = model(u, neg)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


# ============================================================
# 6. Ana akış
# ============================================================

def main():
    print_index_info()
    items_with_pos, negative_pool = print_interactions_and_pools()

    # Model ve optimizer
    model = SimpleMF(NUM_USERS, NUM_ITEMS, EMB_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Başlangıç item embedding'lerini kaydet
    with torch.no_grad():
        init_item_emb = model.item_emb.weight.clone()

    # Train
    for epoch in range(50):
        u, pos, neg = sample_batch(
            batch_size=16,
            train_interactions=TRAIN_INTERACTIONS,
            negative_pool=negative_pool
        )
        loss = bpr_loss(model, u, pos, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Son embedding'leri al
    with torch.no_grad():
        final_item_emb = model.item_emb.weight.clone()

    # Farkları yazdır
    print("\n==== ITEM EMBEDDING FARKLARI ====")
    for i in range(NUM_ITEMS):
        diff = (final_item_emb[i] - init_item_emb[i]).norm().item()
        print(f"Item {i}: embedding değişim normu = {diff:.6f}")

    print("\nYORUM:")
    print("• Item 0–6: Pozitif etkileşimlerde kullanıldığı için yüksek değişim.")
    print("• Item 9 : Pozitif etkileşimi yok ama NEGATIF örnek havuzunda olduğu için")
    print("           loss'ta s(u,neg) skoruna giriyor ve embedding'i DEĞİŞİR (diff > 0).")
    print("• Item 7 ve 8: Ne pozitif ne negatif örnek olarak hiç kullanılmadığı için")
    print("               embedding'leri ilk init değerinde kalır (diff ≈ 0).")
    print("  → Saf MF'de, loss'a giren her item (pozitif veya negatif) mutlaka gradient alır.")
    print("    Loss'a hiç girmeyen item'lar ise 'görünmez' kalır ve öğrenilmez.")


if __name__ == "__main__":
    main()

"""
lightgcn_cold_item_embedding_check.py

Bu script, LightGCN tarzı propagation kullanan basit bir MF modelinde
hangi item embedding'lerinin güncellendiğini görmek için yazılmıştır.

Özellikle:
- Item 0–6: Pozitif etkileşimleri var → graf içinde komşuluğu var → train'de kullanılıyor.
- Item 9 : Pozitif etkileşimi yok, sadece NEGATIF örnek havuzunda → graf içinde komşusu yok.
- Item 7–8: Ne pozitif etkileşimde var ne negatif havuzda → tamamen "cold" ve kullanılmıyor.

Amaç:
- Propagation (A @ E) varken,
  komşusu olmayan ve sadece negatif seçilen item'ların embedding'leri
  gerçekten güncelleniyor mu, güncellenmiyor mu, bunu görmek.
"""

"""
NOT: NODE INDEKSLERININ AÇIK HARITASI
------------------------------------

Bu script'te user ve item node'ları şu kuralla numaralandırılmıştır:

  User node indeksleri : 0 .. NUM_USERS-1
  Item node indeksleri : NUM_USERS .. NUM_USERS+NUM_ITEMS-1

Açık örnek (NUM_USERS = 10, NUM_ITEMS = 10 için):

  USER NODE'LERI:
      Node 0  -> User 0
      Node 1  -> User 1
      Node 2  -> User 2
      Node 3  -> User 3
      Node 4  -> User 4
      Node 5  -> User 5
      Node 6  -> User 6
      Node 7  -> User 7
      Node 8  -> User 8
      Node 9  -> User 9

  ITEM NODE'LERI:
      Node 10 -> Item 0
      Node 11 -> Item 1
      Node 12 -> Item 2
      Node 13 -> Item 3
      Node 14 -> Item 4
      Node 15 -> Item 5
      Node 16 -> Item 6
      Node 17 -> Item 7
      Node 18 -> Item 8
      Node 19 -> Item 9

Bu eşleştirme sayesinde:
- User 0 ile Item 1 etkileşimi  (0, 11) edge'i anlamına gelir.
- User 5 ile Item 0 etkileşimi (5, 10) edge'i anlamına gelir.

Graph adjacency kurarken bu indeksler kullanılmaktadır.
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
# 4. Graf kurma ve normalize edilmiş adjacency (A_norm) oluşturma
# ============================================================

def build_normalized_adjacency(num_users, num_items, interactions):
    """
    User–item bipartite grafından normalize adjacency matrisi üretir.

    Node indeksleri:
      0 .. num_users-1                  -> userlar
      num_users .. num_users+num_items-1 -> itemlar
    """
    num_nodes = num_users + num_items

    rows = []
    cols = []
    vals = []

    # user–item ve item–user (simetrik) ekle
    for u, i in interactions:
        u_idx = u
        i_idx = num_users + i

        # u -> i
        rows.append(u_idx)
        cols.append(i_idx)
        vals.append(1.0)

        # i -> u
        rows.append(i_idx)
        cols.append(u_idx)
        vals.append(1.0)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)

    # Seyrek adjacency matrisi
    adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))

    # Satır bazlı normalize (degree normalizasyonu)
    deg = torch.sparse.sum(adj, dim=1).to_dense()   # [num_nodes]
    deg_inv = torch.zeros_like(deg)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]

    D_inv = torch.diag(deg_inv)
    # A_norm = adj * D^{-1} (row-normalize)
    A_norm = torch.sparse.mm(adj, D_inv)

    return A_norm


# ============================================================
# 5. Model: SimpleMF + LightGCN tarzı propagation
# ============================================================

class SimpleMFWithPropagation(nn.Module):
    """
    LightGCN'e benzer basit model:

    - Önce graph propagation yapılır (E' = A_norm @ E).
    - Sonra skor = dot( user_prop[u], item_prop[i] ).

    Burada:
      self.user_emb, self.item_emb -> trainable embedding tablosu
      propagate() çıktısı          -> forward'da kullanılan temsil
    """
    def __init__(self, n_users, n_items, dim, adj_matrix):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = dim
        self.adj = adj_matrix  # normalized adjacency (sparse)

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def propagate(self):
        """
        1. user+item embeddinglerini birleştir
        2. A_norm ile çarp -> propagation
        3. tekrar user / item olarak ayır
        """
        users = self.user_emb.weight      # [n_users, d]
        items = self.item_emb.weight      # [n_items, d]
        all_emb = torch.cat([users, items], dim=0)  # [n_users+n_items, d]

        # Bu satır LightGCN’in convolution (message passing) kısmıdır
        # Propagation: E' = A * E
        all_prop = torch.sparse.mm(self.adj, all_emb)  # [num_nodes, d]

        users_prop, items_prop = torch.split(
            all_prop,
            [self.n_users, self.n_items],
            dim=0
        )
        return users_prop, items_prop

    def forward(self, u, i):
        users_prop, items_prop = self.propagate()
        u_e = users_prop[u]      # [batch, d]
        i_e = items_prop[i]      # [batch, d]
        scores = (u_e * i_e).sum(dim=1)  # dot product
        return scores


# ============================================================
# 6. BPR örnekleme ve loss
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
# 7. Ana akış
# ============================================================

def main():
    print_index_info()
    items_with_pos, negative_pool = print_interactions_and_pools()

    # Graf ve normalize adjacency
    A_norm = build_normalized_adjacency(NUM_USERS, NUM_ITEMS, TRAIN_INTERACTIONS)

    # Model
    model = SimpleMFWithPropagation(NUM_USERS, NUM_ITEMS, EMB_DIM, A_norm)
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
    print("\n==== ITEM EMBEDDING FARKLARI (trainable parametreler) ====")
    for i in range(NUM_ITEMS):
        diff = (final_item_emb[i] - init_item_emb[i]).norm().item()
        print(f"Item {i}: embedding değişim normu = {diff:.6f}")

    print("\nYORUM:")
    print("• Item 0–6: Pozitif etkileşimlerde kullanıldığı ve negatifte de seçildiği için değişim yüksek.")
    print("• Item 7 ve 8: Graf içinde edge yok ve loss'a hiç girmiyor → embedding'leri aynen kalıyor (diff ≈ 0).")
    print("• Item 9: Bu versiyonda graf'ta komşusu yok ve propagation sonrası temsili 0 olduğu için,")
    print("           negatif örnek olarak seçilse bile trainable embedding'ine gradient ulaşmıyor (diff ≈ 0).")
    print("  → Propagation, parametreyi tek başına güncellemez; sadece A @ E ile forward temsilini değiştirir.")
    print("    Loss, bu temsil embedding'e bağlı değilse ilgili item hiç öğrenilmez.")


if __name__ == "__main__":
    main()

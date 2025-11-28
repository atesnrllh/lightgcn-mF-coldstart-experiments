#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_lightgcn_train_test_overlap.py

Automatically checks user/item overlap consistency for the three
LightGCN-PyTorch datasets (Gowalla, Yelp2018, Amazon-Book) and
plots the distribution of items-per-user in train and test.

Directory structure assumption:
- This script is inside:  /.../MMR/lightgcn-mF-coldstart-experiments
- LightGCN-PyTorch is at: /.../MMR/LightGCN-PyTorch
- Each dataset is inside:
    /.../MMR/LightGCN-PyTorch/data/<dataset_name>/
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def parse_interactions(path):
    """Parse LightGCN format: user item1 item2 ..."""
    users = set()
    items = set()
    user2items = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            uid = int(parts[0])
            user_items = [int(x) for x in parts[1:]]

            users.add(uid)
            if user_items:
                items.update(user_items)

            user2items[uid] = user_items

    return users, items, user2items


def analyze_train_test(train_path, test_path, dataset_name):
    """Analyze consistency for one dataset and plot distributions."""
    print(f"\n===== DATASET: {dataset_name} =====")
    # print(f"Train: {train_path}")
    # print(f"Test : {test_path}")
    print("-" * 60)

    train_users, train_items, train_u2i = parse_interactions(train_path)
    test_users, test_items, test_u2i = parse_interactions(test_path)

    # Toplam etkileşim sayıları
    n_train_interactions = sum(len(v) for v in train_u2i.values())
    n_test_interactions = sum(len(v) for v in test_u2i.values())

    print(f"# train users        : {len(train_users)}")
    print(f"# test users         : {len(test_users)}")
    print(f"# train items        : {len(train_items)}")
    print(f"# test items         : {len(test_items)}")
    print(f"# train interactions : {n_train_interactions}")
    print(f"# test interactions  : {n_test_interactions}")
    if n_test_interactions > 0:
        ratio = n_train_interactions / n_test_interactions
        print(f"train/test interaction ratio : {ratio:.3f}")
    else:
        print("train/test interaction ratio : undefined (n_test_interactions = 0)")
    print("-" * 60)

    # Test user ⊆ train user ?
    missing_users = test_users - train_users
    print("Test users ⊆ Train users ? ", len(missing_users) == 0)
    print(f"Users in test but NOT in train: {len(missing_users)}")

    # Test item ⊆ train item ?
    missing_items = test_items - train_items
    print("Test items ⊆ Train items ? ", len(missing_items) == 0)
    print(f"Items in test but NOT in train: {len(missing_items)}")

    print("-" * 60)
    print(f"Users ONLY in train : {len(train_users - test_users)}")
    print(f"Users ONLY in test  : {len(test_users - train_users)}")
    print(f"Items ONLY in train : {len(train_items - test_items)}")
    print(f"Items ONLY in test  : {len(test_items - train_items)}")
    print("-" * 60)

    # # -------------------------------------------------------
    # # Kullanıcı başına item sayısı dağılımı - 3 grafikli düzen
    # # -------------------------------------------------------
    # train_degs = [len(train_u2i[u]) for u in train_users]
    # test_degs = [len(test_u2i[u]) for u in test_users]
    #
    # max_deg_all = max(max(train_degs), max(test_degs))
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.gridspec import GridSpec
    #
    # fig = plt.figure(figsize=(12, 8))
    # gs = GridSpec(2, 2, height_ratios=[1, 1.2])  # alt grafiği daha büyük yapıyoruz
    #
    # fig.suptitle(f"{dataset_name}: items-per-user distributions", fontsize=14)
    #
    # # ---- 1) Train (sol üst, 0–100) ----
    # ax_train = fig.add_subplot(gs[0, 0])
    # ax_train.hist(train_degs, bins=50, alpha=0.75, color="steelblue")
    # ax_train.set_title("Train distribution (0-100)")
    # ax_train.set_xlabel("Items per user")
    # ax_train.set_ylabel("Number of users")
    # ax_train.set_xlim(0, 100)
    # ax_train.set_xticks(np.arange(0, 101, 5))
    #
    # # ---- 2) Test (sağ üst, 0–100) ----
    # ax_test = fig.add_subplot(gs[0, 1])
    # ax_test.hist(test_degs, bins=50, alpha=0.75, color="orange")
    # ax_test.set_title("Test distribution (0-100)")
    # ax_test.set_xlabel("Items per user")
    # ax_test.set_ylabel("Number of users")
    # ax_test.set_xlim(0, 100)
    # ax_test.set_xticks(np.arange(0, 101, 5))
    #
    # # ---- 3) Train + Test overlay (alt, FULL WIDTH) ----
    # ax_overlay = fig.add_subplot(gs[1, :])  # full width
    # ax_overlay.hist(train_degs, bins=50, alpha=0.6, label="train", color="steelblue")
    # ax_overlay.hist(test_degs, bins=50, alpha=0.6, label="test", color="orange")
    #
    # # Y eksenini log scale yap
    # ax_overlay.set_yscale("log")
    #
    # ax_overlay.set_title("Train + Test (full range, log-scale)")
    # ax_overlay.set_xlabel("Items per user")
    # ax_overlay.set_ylabel("Number of users")
    # ax_overlay.set_xlim(0, max_deg_all)
    # ax_overlay.legend()
    #
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()


if __name__ == "__main__":
    # Bu dosyanın bulunduğu klasör: lightgcn-mF-coldstart-experiments
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # LightGCN-PyTorch bu klasörün bir üstünde, kardeş klasör:
    repo_root = os.path.join(base_dir, "..", "LightGCN-PyTorch")
    data_root = os.path.join(repo_root, "data")

    # Sadece düzgün formatlı 3 dataset:
    datasets = ["gowalla", "yelp2018", "amazon-book"]

    for ds in datasets:
        data_dir = os.path.join(data_root, ds)
        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")

        if not os.path.exists(train_file):
            print(f"\nERROR: train.txt not found for dataset: {ds}")
            continue
        if not os.path.exists(test_file):
            print(f"\nERROR: test.txt not found for dataset: {ds}")
            continue

        analyze_train_test(train_file, test_file, ds)

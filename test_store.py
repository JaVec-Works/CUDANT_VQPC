# test_store.py
import torch
import numpy as np
import faiss
from store import OrbitStore

print("=== STORE.PY TESTS ===\n")

# ---------------------------------------------------------
# TEST 1: FAISS index — add + search
# ---------------------------------------------------------
print("--- TEST 1: FAISS add/search ---")

store = OrbitStore(db_path="test_orbits.db")

# Create 100 fake orbit embeddings (128-dim)
torch.manual_seed(42)
embeddings = torch.randn(100, 128, dtype=torch.float32)
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

# Add them
for i in range(100):
    store.index.add(embeddings[i].numpy()[None])

print(f"  Index size: {store.index.ntotal}")  # expect 100

# Search: find the 5 closest to embedding[7]
D, I = store.index.search(embeddings[7].numpy()[None], 5)
print(f"  Query: embedding[7]")
print(f"  Top 5: {I[0].tolist()}")
print(f"  Distances: {[round(d, 4) for d in D[0]]}")
print(f"  First result is self? {I[0][0] == 7}")  # expect True
print()

# ---------------------------------------------------------
# TEST 2: GPU brute force (the hot path)
# ---------------------------------------------------------
print("--- TEST 2: GPU brute force ---")

# Simulate 1M ticks on GPU
n_ticks = 1_000_000
ticks = torch.randn(n_ticks, 3, dtype=torch.float32, device="cuda")
k_values = torch.arange(n_ticks, dtype=torch.long, device="cuda")
colors = torch.zeros(n_ticks, dtype=torch.long, device="cuda")
colors[::2] = 1  # even = color 0, odd = color 1

# Load into store
store.ticks = ticks
store.k_values = k_values
store.colors = colors

# Query: find 10 closest ticks to a random point
query = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float32, device="cuda")

import time
t0 = time.perf_counter()
d = (store.ticks - query).norm(dim=1)
top_k = d.topk(10, largest=False)
elapsed = (time.perf_counter() - t0) * 1000

print(f"  Device: {ticks.device}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  1M ticks, 3D, GPU brute force: {elapsed:.2f} ms")
print(f"  Closest tick index: {top_k.indices[0].item()}")
print(f"  Closest distance:   {top_k.values[0].item():.6f}")
print()

# ---------------------------------------------------------
# TEST 3: Filtered query (k range + color)
# ---------------------------------------------------------
print("--- TEST 3: Filtered query ---")

t0 = time.perf_counter()
d = (store.ticks - query).norm(dim=1)
mask = (store.k_values >= 500_000) & (store.k_values <= 501_000) & (store.colors == 0)
d[~mask] = float('inf')
top_k = d.topk(10, largest=False)
elapsed = (time.perf_counter() - t0) * 1000

print(f"  Filter: k∈[500000,501000], color=0")
print(f"  Time: {elapsed:.2f} ms")
print(f"  Results: {top_k.indices[:5].tolist()}")
print(f"  All in range? {all(500_000 <= i <= 501_000 for i in top_k.indices.tolist())}")
print(f"  All color 0?  {all(i % 2 == 1 for i in top_k.indices.tolist())}")
print()

# ---------------------------------------------------------
# TEST 4: SQLite metadata
# ---------------------------------------------------------
print("--- TEST 4: SQLite metadata ---")

store.db.execute(
    "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
    ("test_run_1", "flat", 16, 10000,
     np.array([0.0, 0.0, 1.0]).tobytes(), 1732200000.0)
)
store.db.commit()

cursor = store.db.execute("SELECT * FROM runs WHERE traj_id='test_run_1'")
row = cursor.fetchone()
print(f"  traj_id: {row[0]}")
print(f"  tiling:  {row[1]}")
print(f"  N:       {row[2]}")
print(f"  K:       {row[3]}")
v0 = np.frombuffer(row[4], dtype=np.float64)
print(f"  v0:      {v0.tolist()}")
print()

# ---------------------------------------------------------
# TEST 5: Full pipeline (box → store → query)
# ---------------------------------------------------------
print("--- TEST 5: Full pipeline ---")

from box import Box
from tiling import random_tiling

N = 16
T = random_tiling(N, seed=42)
box = Box(N, T)

v0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
v_final, orbit = box.run(v0, K=1000)

# Extract ticks as tensor
ticks_tensor = torch.stack([h for h, _ in orbit], dim=0)  # (1000, 3)
print(f"  Orbit length: {len(orbit)}")
print(f"  Ticks shape:  {ticks_tensor.shape}")
print(f"  First tick:   {ticks_tensor[0].tolist()}")
print(f"  Last tick:    {ticks_tensor[-1].tolist()}")
print(f"  Final dir:    {v_final.tolist()}")

# Store it
store.ticks = ticks_tensor.cuda()
store.k_values = torch.arange(len(orbit), dtype=torch.long, device="cuda")
store.colors = torch.zeros(len(orbit), dtype=torch.long, device="cuda")

# Query: "which tick is closest to the origin?"
origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
d = (store.ticks - origin).norm(dim=1)
closest = d.argmin().item()
print(f"  Closest tick to origin: k={closest}, pos={ticks_tensor[closest].tolist()}")
print()

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
import os
store.db.close()
os.remove("test_orbits.db")

print("=== ALL TESTS PASSED ===")   
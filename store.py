# CUDANT_VQPC — store.py
# FAISS (128-dim orbit embeddings) + SQLite (metadata + ticks) + GPU brute force (3D)
# No server. No daemon. No network. Same GPU, same process.

import time
import torch
import faiss
import sqlite3
import numpy as np


class OrbitStore:
    def __init__(self, db_path="orbits.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS runs (
            traj_id TEXT PRIMARY KEY,
            tiling TEXT, N INT, K INT, v0 BLOB, timestamp REAL
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS ticks (
            id TEXT PRIMARY KEY,
            x REAL, y REAL, z REAL,
            k INTEGER,
            color INTEGER,
            tiling TEXT,
            timestamp REAL
        )""")
        self.db.commit()
        self.index = faiss.IndexFlatIP(128)
        self.ticks = None
        self.k_values = None
        self.colors = None

    def store_orbit(self, traj_id, orbits, tiling_version, N, K, ticks_every=100):
        """Store sampled ticks from box.run() orbits. Called by run_box."""
        ids = []
        for color_idx, orbit in enumerate(orbits):
            for k in range(0, len(orbit), ticks_every):
                h, tile = orbit[k]
                x, y, z = h[0].item(), h[1].item(), h[2].item()
                pid = f"{traj_id}_{color_idx}_{k}"
                self.db.execute(
                    "INSERT OR REPLACE INTO ticks VALUES (?,?,?,?,?,?,?,?)",
                    (pid, x, y, z, k, color_idx, tiling_version, time.time())
                )
                ids.append(pid)
        self.db.commit()
        return ids

    def store_run(self, traj_id, tiling, N, K, v0):
        """Store run metadata."""
        self.db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
            (traj_id, tiling, N, K, np.array(v0).tobytes(), time.time())
        )
        self.db.commit()

    def add_embedding(self, embedding):
        """Add a 128-dim orbit embedding to the FAISS index."""
        self.index.add(embedding.cpu().numpy().astype(np.float32)[None])

    def load_ticks(self, ticks_gpu, k_values=None, colors=None):
        """Load ticks into memory for GPU brute-force queries."""
        self.ticks = ticks_gpu
        self.k_values = k_values
        self.colors = colors

    def query_point(self, p, k_range=None, color=None, limit=10):
        """Brute force on GPU. µs."""
        d = (self.ticks - p).norm(dim=1)
        if k_range and self.k_values is not None:
            mask = (self.k_values >= k_range[0]) & (self.k_values <= k_range[1])
            d[~mask] = float('inf')
        if color is not None and self.colors is not None:
            d[self.colors != color] = float('inf')
        return d.topk(limit, largest=False)

    def query_similar(self, embedding, limit=10):
        """ANN on 128-dim embeddings. ms."""
        D, I = self.index.search(
            embedding.cpu().numpy().astype(np.float32)[None], limit)
        return I[0].tolist()

    def find_attractors(self, radius=0.01, min_hits=100):
        """Placeholder: spatial clustering of ticks."""
        return []

    def close(self):
        """Close the database connection (needed before deleting the file on Windows)."""
        self.db.close()   
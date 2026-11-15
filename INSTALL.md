
# CUDANT_VQPC — INSTALL

**Zero to hero. Any level. Windows-first (Linux/Mac notes included).**

---

## What you need

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04+ | Windows 11 |
| Python | 3.10 | 3.12 |
| GPU | NVIDIA, 4 GB VRAM | NVIDIA, 8 GB VRAM |
| CUDA | 11.8 | 12.x+ |
| RAM | 16 GB | 32 GB+ |
| Disk | 2 GB (PyTorch) | 5 GB |

**Check your GPU:**

```powershell
nvidia-smi
```

If it prints your GPU name and CUDA version, you're good. If it says "N-SMH" or "not found," install the NVIDIA driver first: [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers)

---

## Step 1: Install Python

If you don't have Python 3.12:

1. Go to https://python.org/downloads
    
2. Download Python 3.12.x
    
3. Run the installer. **Check "Add to PATH"** on the first screen.
    
4. Click "Install Now."
    

Verify:

```
python --version
# Should print: Python 3.12.x
```

---

## Step 2: Get the code

```
git clone <repo-url>
cd CUDANT_VQPC
```

---

## Step 3: Create the environment

```
python -m venv .venv
```

**Windows:**

```
.venv\Scripts\activate
```

**Linux/Mac:**

```
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt. You're now inside the project's isolated environment.

---

## Step 4: Install dependencies

```
pip install -r requirements.txt
```

This installs:

- `torch` (PyTorch with CUDA — ~2.5 GB download)
    
- `numpy`
    
- `fastmcp` (Model Context Protocol server)
    
- `matplotlib` (for the visualizer)
    

**Verify CUDA is working:**

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should print:

```
True
NVIDIA Quadro RTX 4000   (or your GPU)
```

If it prints `False`, your PyTorch install doesn't have CUDA. Reinstall:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 5: Generate settings

```
python benchmark.py
```

This:

- Detects your GPU
    
- Tests VRAM usage
    
- Writes `settings.conf` (VRAM cap, max lions)
    

You should see output like:

```
  VRAM cap: 5.8% (500 MB for CUDANT_VQPC)
  Max lions: 1024
```

---

## Step 6: Run the tests

```
python particle.py      # The particle. Watch it rotate.
python gates.py         # All gates. Watch them work.
python circuit.py       # Full pipeline. Watch the wobble.
python cage.py          # Multi-particle. Watch entanglement.
python engine.py        # 16-qubit exact. Watch Grover.
python algorithms.py    # All 5 algorithms.
python learn.py         # The particle learns.
python visualize.py     # Generates 3 GIFs.
```

Each one should print "✅" for every test. If any fails, check the error message.

---

## Step 7: Run the server

```
python vqpc_server.py
```

You should see:

```
  VRAM cap: 5.8% (500 MB for CUDANT_VQPC)
  Max lions: 1024
=== CUDANT-VQPC MCP SERVER ===
Tools: run_circuit, measure_state, wobble_report, list_gates, run_cage, run_engine, run_algorithm, train_particle, replay, help
Max lions: 1024 | VRAM cap: 6%
Connecting via stdio...
```

It's now waiting for an agent to connect via MCP. **Ctrl+C** to stop.

---

## Step 8: Connect an agent

### Claude Desktop

1. Open Claude Desktop → Settings → Developer → Edit `claude_desktop_config.json`
    
2. Add:
    

```
{
  "mcpServers": {
    "cudant-vqpc": {
      "command": "python",
      "args": ["D:\\PROJECTS\\CUDANT_VQPC\\vqpc_server.py"],
      "cwd": "D:\\PROJECTS\\CUDANT_VQPC"
    }
  }
}
```

3. Restart Claude Desktop.
    
4. In a new chat, the agent will see 10 new tools. Ask it: _"Call help on the quantum computer."_
    

### Other MCP clients

Any MCP-compatible client works. The server speaks stdio. Point it at `python vqpc_server.py` with the working directory set to the project folder.

---

## Step 9: Open the GIFs

After running `python visualize.py`, three GIFs appear in the project folder:

- `replay_clean.gif` — straight line (single rotation)
    
- `replay_wobbly.gif` — cone (10 small rotations)
    
- `replay_kickback.gif` — curve (H, T, H phase kickback)
    

Open them in Windows Photos, or any image viewer. The red arrow is the particle. The orange trail is its path.

---

## Troubleshooting

|   |   |
|---|---|
|Problem|Fix|
|`torch.cuda.is_available()` returns `False`|Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`|
|`settings.conf not found`|Run `python benchmark.py` first|
|`ModuleNotFoundError: fastmcp`|`pip install fastmcp` (inside the venv)|
|`ModuleNotFoundError: matplotlib`|`pip install matplotlib` (inside the venv)|
|GPU not detected|Update NVIDIA driver. Check `nvidia-smi` in a fresh terminal.|
|`&&` not recognized in PowerShell|Use `;` instead (PowerShell 5.1 doesn't support `&&`)|
|`.venv` in git|It shouldn't be. Check `.gitignore` has `.venv/`|
|VRAM error|Lower the cap in `settings.conf` → `[gpu] memory_fraction`|

---

## Uninstall

```
Remove-Item -Recurse -Force .venv
```

That's it. The `.venv` folder is the entire installation. Delete it and you're back to zero.

---

## Updating

```
git pull
pip install -r requirements.txt
python benchmark.py
```


---
EOF

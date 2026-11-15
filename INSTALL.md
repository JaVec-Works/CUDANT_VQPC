# INSTALL — CUDANT_VQPC

**From zero to "TRUSTED" in 10 minutes. No experience needed.**

## What you need

|   |   |   |
|---|---|---|
|Thing|Minimum|Where to get it|
|Windows 10 or 11|Yes|You already have it|
|NVIDIA GPU|4 GB VRAM (GTX 1050 or better)|You already have it|
|Python|3.12|python.org|
|Internet|For one `pip install`|You already have it|

That's it. No cloud account. No quantum hardware. No CUDA toolkit install. No cryogenics.

---

## Step 1: Install Python (skip if you already have it)

1. Go to **python.org/downloads**
    
2. Click **"Download Python 3.12.x"**
    
3. Run the installer
    
4. **IMPORTANT:** Check the box that says **"Add Python to PATH"**
    
5. Click **"Install Now"**
    
6. Close the installer
    

**Verify:** Press `Win + X`, type "PowerShell", hit Enter. Then type:

```
python --version
```

You should see: `Python 3.12.x`

If you see an error, restart your computer and try again.

---

## Step 2: Get the code

**Option A — No Git (easiest):**

1. Open a browser, go to the repo URL
    
2. Click the green **"Code"** button → **"Download ZIP"**
    
3. Unzip it to a folder you like, e.g. `D:\PROJECTS\CUDANT_VQPC`
    

**Option B — With Git:**

Open PowerShell and type:

```
git clone <repo-url> D:\PROJECTS\CUDANT_VQPC
```

---

## Step 3: Make a virtual environment

Open PowerShell. Navigate to your folder:

```
cd D:\PROJECTS\CUDANT_VQPC
```

Create the virtual environment:

```
python -m venv .venv
```

Activate it:

```
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of your prompt:

```
(.venv) PS D:\PROJECTS\CUDANT_VQPC>
```

**If you get an error about "running scripts is disabled":**

Type this first, then try the activate command again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Type `Y` and press Enter when it asks for confirmation.

---

## Step 4: Install the packages

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy fastmcp
```

**Important:** The first command installs PyTorch with CUDA support. If you use plain `pip install torch`, you'll get the CPU-only version and the GPU won't work.

**Verify your GPU is detected:**

```
python -c "import torch; print(torch.cuda.is_available())"
```

You should see: `True`

If you see `False`, your NVIDIA drivers are outdated. Update them at **[nvidia.com/drivers](https://nvidia.com/drivers)**, then try again.

**If your CUDA version is 12.1 instead of 12.4:**

Check with `nvidia-smi` in a separate PowerShell window (look at the top-right "CUDA Version"). If it says 12.1, use:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5: Run it

**The particle (the core):**

```
python particle.py
```

You should see the particle rotate to any point on the sphere, wobble in a cone, hit the cage (measurement), and 1000 particles computed in one batch.

**The 5 laws (the physics engine):**

```
python laws.py
```

You should see superposition, reversibility, entanglement, collapse, and interference demonstrated.

**The gates (how you steer it):**

```
python gates.py
```

You should see each gate move the particle, H·H=I, S·S†=I, and 1000 particles in one batch.

**Measurement (looking at the particle):**

```
python measure.py
```

You should see single-shot collapse and 1000-shot sampling matching theory.

**The full pipeline (the virtual quantum computer):**

```
python circuit.py
```

You should see circuits running with trajectory, wobble detection, and measurement. The "Clean vs Wobbly" test shows the agent can see the difference between a straight path and a cone.

**The AI agent server:**

```
python vqpc_server.py
```

This starts the MCP server. It won't print anything useful on its own — it waits for an AI agent to connect. You'll see:

```
=== CUDANT-VQPC MCP SERVER ===
Tools: run_circuit, measure_state, wobble_report, list_gates
Connecting via stdio...
```

---

## Step 6: Connect an AI agent (optional)

If you have an MCP-capable client (PI harness, Claude Desktop, etc.), add this to your config:

```
{
  "vqpc": {
    "command": "D:\\PROJECTS\\CUDANT_VQPC\\.venv\\Scripts\\python.exe",
    "args": ["D:\\PROJECTS\\CUDANT_VQPC\\vqpc_server.py"],
    "cwd": "D:\\PROJECTS\\CUDANT_VQPC",
    "exposure": "direct"
  }
}
```

Adjust the path to wherever you put the project.

Then ask your agent:

> "Use the vqpc server. Call the run_circuit tool with gates='H, T, H', n_shots=1000. Report the tool output exactly as returned. Do not call any other tools."

It will run the circuit, measure it, check the wobble, and report back with a verdict.

---

## Troubleshooting

|   |   |
|---|---|
|Problem|Fix|
|`python` not recognized|Reinstall Python, check "Add to PATH", restart computer|
|`(.venv)` not appearing|You're in the wrong folder. Type `cd D:\PROJECTS\CUDANT_VQPC`|
|`torch.cuda.is_available()` = False|Update NVIDIA drivers at [nvidia.com/drivers](https://nvidia.com/drivers)|
|"Torch not compiled with CUDA enabled"|Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu124`|
|`ModuleNotFoundError: No module named 'fastmcp'`|`pip install fastmcp`|
|Circular import error with fastmcp|Make sure your server file is NOT named `mcp.py`. Use `vqpc_server.py`.|
|Out of memory (OOM) error|Close other GPU apps. Single-particle VQPC needs very little VRAM.|
|"Running scripts is disabled"|Type `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, press Y|
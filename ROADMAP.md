# ROADMAP — CUDANT_VQPC

**The virtual quantum computer. Built in order. Checked off as we go.**

Last updated: 2026-11-14

---

## PHASE 0: FOUNDATION (DONE)

The particle, the laws, the gates, the measurement, the circuit, the cage.

- [x] `particle.py` — Full Bloch sphere (θ + φ), trajectory, measurement
    
- [x] `laws.py` — 5 quantum laws (physics engine)
    
- [x] `gates.py` — I, X, Y, Z, H, S, S†, T, T† + Rx(θ), Ry(θ), Rz(θ)
    
- [x] `measure.py` — Single-shot collapse + N-shot sampling
    
- [x] `circuit.py` — Full pipeline: gates → state → trajectory → wobble → measurement
    
- [x] `cage.py` — N particles, rods (CNOT), entanglement, correlated measurement
    
- [x] `vqpc_server.py` — 6 MCP tools (run_circuit, measure_state, wobble_report, list_gates, run_cage, help)
    
- [x] `benchmark.py` — Spec sheet + settings.conf generation
    
- [x] `settings.conf` — VRAM cap (500 MB), max lions (1024), timing reference
    
- [x] `requirements.txt` — torch, numpy, fastmcp
    
- [x] `README.md` — Project description
    
- [x] `INSTALL.md` — Zero-to-hero guide
    
- [x] GitHub repo (private) — pushed, clean, no .venv
    
- [x] PI harness wiring — 5/5 tools tested clean with Huihui
    

---

## PHASE 1: AXIS CONVENTION (PENDING)

**The problem:** Our rotation axes work, but the convention is inconsistent across files. After `Rx(θ)`, the vector lands in the Y-Z plane and φ reads as -90°. After `Ry(θ)`, it lands in the X-Z plane and φ reads as 0°. The user's design says "any angle, any direction, any elevation" — the math is correct, but the _labels_ (what θ and φ mean after each rotation) need to be locked down and documented.

**What to do:**

- [ ] Write a 1-page `AXIS.md` that defines:
    
    - θ = elevation from north pole (|0>). 0° = up, 180° = down.
        
    - φ = direction around the equator, measured from +X axis.
        
    - Rx(θ): rotates in Y-Z plane. After Rx, φ = -90° (vector is along -Y).
        
    - Ry(θ): rotates in X-Z plane. After Ry, φ = 0° (vector is along +X).
        
    - Rz(φ): rotates in X-Y plane. Changes φ only. θ unchanged.
        
    - H = Ry(90°) followed by Rz(90°). (Or: the specific 3×3 matrix we use.)
        
    - The "correlation axis" for entangled pairs is +X. Document why.
        
- [ ] Add a comment block at the top of `particle.py` referencing `AXIS.md`
    
- [ ] Add a comment block at the top of `gates.py` referencing `AXIS.md`
    
- [ ] Verify: after `Ry(60)`, φ should read 0°. After `Rx(60)`, φ should read -90°. After `Rz(45)` on a vector already at φ=0°, φ should read 45°.
    
- [ ] Update `README.md` "The 5 Laws" section to reference the axis convention
    

**Why it matters:** If the agent calls `Ry(60)` expecting "tilt down 60°, stay facing the same direction," it gets that. But if it calls `Rx(60)` expecting the same, it gets "tilt down 60°, now facing -Y." The agent needs to know which rotation does what without guessing.

---

## PHASE 2: MERGE VQPC + CUDANT (PENDING)

**The goal:** VQPC (dashboard) calls CUDANT (engine) for true multi-qubit computation.

**What to do:**

- [ ] Create `engine.py` in CUDANT_VQPC that imports CUDANT's `lion16.py` gate functions
    
- [ ] Add a 7th MCP tool: `run_engine(n_qubits, gates, n_shots)` — calls the 16-qubit state vector engine
    
- [ ] The agent's workflow becomes:
    
    - Simple (1-2 qubits): use `run_circuit` or `run_cage` (geometry, fast, intuitive)
        
    - Complex (3-16 qubits): use `run_engine` (exact state vector, full entanglement)
        
    - The agent decides which tool based on the problem
        
- [ ] Add a `help()` update that explains when to use which tool
    
- [ ] Test: run a 4-qubit GHZ state through `run_engine`, verify P(0000) and P(1111) = 0.5 each
    
- [ ] Test: run the same through `run_cage` (4 lions, 3 rods), compare results
    
- [ ] Document the difference: cage = geometric approximation, engine = exact
    

**Files to touch:**

- NEW: `engine.py` (wraps CUDANT's lion16.py)
    
- EDIT: `vqpc_server.py` (add 7th tool, update help)
    
- EDIT: `README.md` (architecture section: dashboard + engine)
    

---

## PHASE 3: USEFUL ALGORITHMS (PENDING)

**The goal:** The agent can run real quantum algorithms, not just arbitrary gates.

**What to do:**

- [ ] `algorithms.py` — pre-built circuits as functions:
    
    - [ ] `grover(n_qubits, target)` — search. Uses engine (needs full state vector).
        
    - [ ] `bell_pair(n_qubits)` — entanglement demo. Works in cage.
        
    - [ ] `ghz_state(n_qubits)` — multi-qubit entanglement. Needs engine for n>2.
        
    - [ ] `quantum_fourier_transform(n_qubits)` — QFT. Needs engine.
        
    - [ ] `phase_estimation(n_qubits, eigenvalue)` — Needs engine.
        
- [ ] Add an 8th MCP tool: `run_algorithm(name, params)` — calls the right function
    
- [ ] Each algorithm returns: answer + trajectory (from VQPC) + exact probabilities (from engine) + verdict
    
- [ ] Test: Grover on 4 qubits, target = 0b1010. Should find it in ~4 iterations.
    
- [ ] Test: Bell pair in cage vs. engine. Results should match.
    

**Why this is the "useful" step:** Right now the agent can run _any_ circuit. But it has to know _what circuit to run_ for a given problem. Algorithms package that knowledge. "Find item X in an unsorted database" → Grover. "Estimate the energy of a molecule" → phase estimation. The agent says the problem, the tool picks the algorithm.

---

## PHASE 4: LEARNING LAYER (PENDING)

**The goal:** Merge CUDANT-DL (the learnable layer from the first chat) into VQPC.

**What to do:**

- [ ] Port `lion16_layer.py` logic into VQPC as `learn.py`
    
- [ ] The particle's angles become `nn.Parameter`s (learnable)
    
- [ ] Add a 9th MCP tool: `train_particle(target, steps, lr)` — trains the particle to point at a target
    
- [ ] The wobble becomes a regularizer: "don't just reach the target, reach it cleanly"
    
- [ ] Test: train the particle to point at θ=60°, φ=45°. Verify it converges.
    
- [ ] Test: train with wobble penalty. Verify the path is straighter than without.
    

**Why this matters:** This is the "the particle learns" step. The quantum layer that CUDANT-DL proved works, now living inside the VQPC architecture with the trajectory and wobble as training signals.

---

## PHASE 5: CUDANT-128-DL (FUTURE)

**The goal:** Differentiable 128-qubit layer via cuTensorNet.

**Blocked on:** CUDA-Q ML open-source release (estimated Q1-Q2 2027).

**What to do (when unblocked):**

- [ ] Watch `cuda-quantum` GitHub for `cudaq-ml` package
    
- [ ] Wrap CUDA-Q MPS in `torch.autograd.Function`
    
- [ ] Forward: CUDA-Q circuit. Backward: `cutensornetNetworkComputeGradientsBackward`
    
- [ ] Two 64-qubit blocks, ECC consistency term
    
- [ ] Run the same `CONVERGED` test at 64 qubits
    
- [ ] Add to VQPC as a 10th tool: `run_128(n_qubits, gates, chi)`
    

---

## PHASE 6: COMMERCIAL (FUTURE)

**The goal:** Make money.

**Steps (in order):**

- [ ] **Paper** (after Phase 2): "Agent-Drivable Quantum Layer with Trajectory Diagnostics on Consumer GPU." Submit to a quantum computing or AI conference.
    
- [ ] **Demo video** (after paper): 10 minutes. Show the agent running Grover, seeing the wobble, getting the verdict. Post on YouTube.
    
- [ ] **Open CUDANT-128** (after paper): Make the public repo the "scale" story. CUDANT + CUDANT_VQPC stay private (the product).
    
- [ ] **Education** (6-12 months): "Quantum Computing on a $500 GPU" course. Universities, bootcamps.
    
- [ ] **Hosted API** (12-18 months): "Run quantum circuits without installing anything." $20/mo.
    
- [ ] **MCP marketplace** (when it matures): "The quantum tool for your AI agent."
    

---

## KNOWN ISSUES (TRACKING)

|   |   |   |   |   |
|---|---|---|---|---|
|#|Issue|Where|Severity|Status|
|1|Axis convention inconsistent (Rx gives φ=-90°, Ry gives φ=0°)|particle.py, gates.py|Medium|Phase 1|
|2|H gate matrix is a specific 3×3, not a standard "rotation by 45°"|gates.py|Low|Phase 1 (document)|
|3|Entangled pair snaps both vectors to [1,0,0] (simplification)|cage.py|Medium|Document in AXIS.md. Full fix needs engine.|
|4|Wobble threshold for "CAUTION" too aggressive for phase gates|vqpc_server.py|Low|Tune after Phase 1|
|5|`run_cage` caps at 8 lions in tool docstring (settings.conf says 1024)|vqpc_server.py|Low|Already fixed (reads _max_lions)|
|6|cage.py header prints twice (top-level + inside guard)|cage.py|Cosmetic|Fix when touching file|
|7|Test 4 in particle.py: 10 shots is too few for 50/50 demo|particle.py|Cosmetic|Change to 100 shots|

---

## FILE TREE (CURRENT)

```
D:\PROJECTS\CUDANT_VQPC\
├── particle.py          ✅
├── laws.py              ✅
├── gates.py             ✅
├── measure.py           ✅
├── circuit.py           ✅
├── cage.py              ✅
├── vqpc_server.py       ✅ (6 tools, VRAM cap)
├── benchmark.py         ✅
├── settings.conf        ✅ (auto-generated)
├── requirements.txt     ✅
├── README.md            ✅
├── INSTALL.md           ✅
├── ROADMAP.md           ← this file
├── .gitignore           ✅
├── .venv/               (excluded from git)
│
├── [Phase 1] AXIS.md    ← pending
├── [Phase 2] engine.py  ← pending
├── [Phase 3] algorithms.py ← pending
├── [Phase 4] learn.py   ← pending
└── [Phase 5] (CUDANT-128-DL, separate repo)
```

---

## RULES

1. **Never break a working file.** If you edit `particle.py`, run `python particle.py` after.
    
2. **One phase at a time.** Don't start Phase 2 until Phase 1 is checked off.
    
3. **Commit after every phase.** `git add . && git commit -m "Phase N: description" && git push`
    
4. **The axis convention is law.** Once `AXIS.md` is written, every file follows it. No more "φ = -90° surprise."
    
5. **The agent is the user.** If the agent can't figure out how to use a tool from the docstring alone, the docstring is wrong.
    

---

_Built by two people who believe a $500 GPU is a quantum computer wearing a different costume._
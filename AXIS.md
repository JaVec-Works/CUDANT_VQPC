# AXIS CONVENTION — CUDANT_VQPC

**The law. Every file follows this. No more "φ = -90° surprise."**

---

## The Sphere

```
        Z (up = |0>)
        |
        |
  ------+------  X (right = +X, "correlation axis")
       /|
      / |
     Y  |

θ (elevation): angle from the north pole (Z).
  0°   = pointing up   = |0>
  90°  = pointing sideways (equator)
  180° = pointing down = |1>

φ (direction): angle around the equator, measured from +X axis, counter-clockwise looking down from +Z.
  0°   = pointing along +X
  90°  = pointing along +Y
  180° = pointing along -X
  270° = pointing along -Y
```

**Bloch vector from (θ, φ):**

```
x = sin(θ) · cos(φ)
y = sin(θ) · sin(φ)
z = cos(θ)
```

**Probabilities from the vector:**

```
P(|0>) = (1 + z) / 2
P(|1>) = (1 - z) / 2
```

Note: φ does NOT affect P(|0>) or P(|1>). Only z matters for measurement. φ is the "hidden" phase — invisible until you interfere.

---

## The Rotations

### Rx(θ) — Rotate around X axis

**Matrix:**

```
[1,  0,       0     ]
[0,  cos(θ), -sin(θ)]
[0,  sin(θ),  cos(θ)]
```

**Effect on the vector:** Swings in the **Y-Z plane**. X stays fixed.

**Starting from |0> = [0, 0, 1]:**

```
After Rx(θ): [0, -sin(θ), cos(θ)]
```

|   |   |   |   |   |
|---|---|---|---|---|
|θ|Vector|θ_elev|φ|Notes|
|0°|[0, 0, 1]|0°|0°|Still|0>|
|45°|[0, -0.707, 0.707]|45°|-90°|Tilted, facing -Y|
|90°|[0, -1, 0]|90°|-90°|Sideways, facing -Y|
|180°|[0, 0, -1]|180°|undefined|Now|1> (pole)|

**THE GOTCHA:** After Rx, **φ = -90°** (not 0°). The vector swings into the -Y direction. If you want "tilt down but keep facing +X," use **Ry**, not Rx.

---

### Ry(θ) — Rotate around Y axis

**Matrix:**

```
[cos(θ), 0, sin(θ)]
[0,      1, 0     ]
[-sin(θ),0, cos(θ)]
```

**Effect on the vector:** Swings in the **X-Z plane**. Y stays fixed.

**Starting from |0> = [0, 0, 1]:**

```
After Ry(θ): [sin(θ), 0, cos(θ)]
```

|   |   |   |   |   |
|---|---|---|---|---|
|θ|Vector|θ_elev|φ|Notes|
|0°|[0, 0, 1]|0°|0°|Still|0>|
|45°|[0.707, 0, 0.707]|45°|0°|Tilted, facing +X|
|90°|[1, 0, 0]|90°|0°|Sideways, facing +X (= H)|
|180°|[0, 0, -1]|180°|undefined|Now|1> (pole)|

**THE INTUITIVE ONE:** After Ry, **φ = 0°**. "Tilt down, keep facing the same direction." This is the rotation to use when you want elevation change without direction change.

---

### Rz(φ) — Rotate around Z axis

**Matrix:**

```
[cos(φ), -sin(φ), 0]
[sin(φ),  cos(φ), 0]
[0,       0,      1]
```

**Effect on the vector:** Spins in the **X-Y plane**. Z stays fixed. **θ does not change.**

**Starting from any vector with θ ≠ 0°:**

```
After Rz(Δφ): φ_new = φ_old + Δφ
```

|   |   |   |
|---|---|---|
|Starting φ|After Rz(45°)|Notes|
|0°|45°|Spun 45° counter-clockwise|
|-90°|-45°|Spun 45° counter-clockwise|
|90°|135°|Spun 45° counter-clockwise|

**THE INVISIBLE GATE:** On |0> (θ=0°), Rz does **nothing visible**. The vector is at the pole. Spinning the pole doesn't move it. The phase is "hidden." It only shows up when the particle is in superposition (θ ≠ 0°).

---

## The Named Gates (in terms of rotations)

| Gate | Equivalent rotation | Effect on |0> | Resulting (θ, φ) |  
|------|-------------------|-----------|---------------------|  
| I | None | [0,0,1] | (0°, 0°) |  
| X | Rx(180°) | [0,0,-1] | (180°, undefined) |  
| Y | Ry(180°) | [0,0,-1] | (180°, undefined) |  
| Z | Rz(180°) | [0,0,1] | (0°, 0°) — invisible on |0> |  
| H | **Ry(90°)** | [1,0,0] | (90°, 0°) |  
| S | Rz(90°) | [0,0,1] | (0°, 0°) — invisible on |0> |  
| T | Rz(45°) | [0,0,1] | (0°, 0°) — invisible on |0> |

**Key insight:** Z, S, T on |0> all look identical (no movement). They only matter on superposition. This is by design — it's how phase kickback works (H → T → H).

---

## The H Gate (specifically)

Our H matrix:

```
[0, 0, 1]
[0, 1, 0]
[1, 0, 0]
```

**H = Ry(90°).** Verified:

- H|0> = [1, 0, 0] = Ry(90°)|0> ✓
    
- H|1> = [-1, 0, 0] = Ry(90°)|1> ✓
    
- H|+> = |0> ✓ (H is its own inverse)
    

After H, the particle is at **(θ=90°, φ=0°)** — on the equator, facing +X.

---

## The Correlation Axis (Entanglement)

When two particles are entangled (CNOT on a superposed control), we snap both vectors to **[1, 0, 0]** — the **+X axis**.

**Why +X?**

- After H (= Ry(90°)), the control is at [1, 0, 0] (facing +X).
    
- The Bell state |Φ+> = (|00>+|11>)/√2 has its correlation along X.
    
- Measuring along Z: each particle shows 50/50. But they always agree.
    
- The +X direction is the "we're linked" direction. It's the axis along which the correlation lives.
    

**In code:** `cage.py` sets both entangled particles to `[1.0, 0.0, 0.0]`. This is a **simplification** (the true reduced state of a Bell pair qubit is the maximally mixed state = origin, not a vector on the sphere). But it's the GPU-native way to say "these two vertices are one object, correlated along X." The measurement logic enforces the correlation.

---

## Quick Reference: "I want to..."

|   |   |   |
|---|---|---|
|I want to...|Use|Result|
|Tilt down, keep facing same direction|**Ry(θ)**|θ changes, φ stays 0°|
|Tilt down, face a new direction|**Rx(θ)**|θ changes, φ becomes -90°|
|Spin in place (change direction only)|**Rz(φ)**|φ changes, θ stays|
|Create superposition (50/50)|**H** (= Ry(90°))|θ=90°, φ=0°|
|Flip|0> to|1>|**X** (= Rx(180°))|θ=180°|
|Add a phase (invisible on|0>)|**Z, S, T**|No visible change on|0>|
|Go to any point on the sphere|**Ry(θ) then Rz(φ)**|First tilt, then spin|

**The universal recipe for "point the particle at (θ, φ)":**

```
1. Start at |0> (θ=0°, φ=0°)
2. Ry(θ)  → tilt to the right elevation (now at θ, φ=0°)
3. Rz(φ)  → spin to the right direction (now at θ, φ)
```

Two gates. Anywhere on the sphere. That's the full freedom your design called for.

---

## File Cross-Reference

|   |   |
|---|---|
|File|Where axes matter|
|`particle.py`|`_to_vector()` converts (θ,φ)→(x,y,z). `rotate_x/y/z()` apply the matrices. `_update_angles()` recovers (θ,φ) from (x,y,z).|
|`gates.py`|`rotation_x/y/z()` build the 3×3 matrices. `GATES` dict maps names to specific rotations.|
|`cage.py`|`apply_cnot()` snaps entangled pairs to [1,0,0] (+X correlation axis).|
|`circuit.py`|`run_circuit()` parses "Rx(30)", "Ry(45)", "Rz(60)" and applies them.|
|`vqpc_server.py`|Tool docstrings must reference this convention.|

---

## Rules

1. **Ry is the "tilt" gate.** If you want to change elevation without changing direction, use Ry.
    
2. **Rx is the "tilt + turn" gate.** After Rx, you're facing -Y. Know this.
    
3. **Rz is the "spin" gate.** Changes direction only. Invisible on poles.
    
4. **H = Ry(90°).** Always. Not "some diagonal thing." It's a 90° tilt around Y.
    
5. **The correlation axis is +X.** Entangled pairs live on the +X axis. Document it, don't guess it.
    
6. **φ is invisible in measurement.** P(|0>) and P(|1>) depend only on z. φ shows up in interference.
    
7. **At the poles (θ=0° or 180°), φ is undefined.** We set it to 0° by convention. Don't read meaning into it.
    

---

_This file is the law. If a test output contradicts this document, the code is wrong, not the document._
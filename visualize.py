# =============================================================================
# CUDANT_VQPC — visualize.py
# Sidecar visualizer. Not imported by default. Called by the replay tool.
# Generates a 3D Bloch sphere GIF with the vector animating through gates.
#
# Run: python visualize.py  (generates a demo GIF)
# =============================================================================

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from circuit import run_circuit
from particle import Particle
from gates import GATES, apply_gate


def _build_trajectory(gates_str: str) -> list:
    """
    Run the circuit and return a list of (label, vector) at each step.
    """
    p = Particle(theta=0.0, phi=0.0)
    trajectory = [("Start |0>", p.v.clone().cpu().numpy())]
    
    for g in gates_str.split(","):
        g = g.strip()
        if not g:
            continue
        
        if "(" in g and g.endswith(")"):
            name = g[:g.index("(")]
            angle_deg = float(g[g.index("(")+1:-1])
            angle = np.radians(angle_deg)
            if name == "Rx":
                p.rotate_x(angle)
            elif name == "Ry":
                p.rotate_y(angle)
            elif name == "Rz":
                p.rotate_z(angle)
            label = g
        else:
            R = GATES[g]
            p.v = R @ p.v
            p.v = p.v / p.v.norm()
            p._update_angles()
            label = g
        
        trajectory.append((label, p.v.clone().cpu().numpy()))
    
    return trajectory


def _interpolate(v1: np.ndarray, v2: np.ndarray, n_frames: int) -> list:
    """
    Interpolate between two vectors on the sphere (slerp).
    Returns n_frames intermediate vectors.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dots = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dots)
    
    if theta < 1e-6:
        return [v1.copy() for _ in range(n_frames)]
    
    frames = []
    for i in range(1, n_frames + 1):
        t = i / n_frames
        a = np.sin((1 - t) * theta) / np.sin(theta)
        b = np.sin(t * theta) / np.sin(theta)
        frames.append(a * v1 + b * v2)
    
    return frames


def generate_gif(gates_str: str, slowmo: bool = True, output_path: str = "replay.gif") -> dict:
    """
    Generate a Bloch sphere animation GIF.
    
    Args:
        gates_str: Gate sequence (same format as run_circuit).
        slowmo: If True, interpolate between gates (smoother animation).
                If False, one frame per gate (jumpy).
        output_path: Where to save the GIF.
    
    Returns:
        dict with file path, frame count, trajectory data.
    """
    trajectory = _build_trajectory(gates_str)
    frames = []
    
    for i, (label, v) in enumerate(trajectory):
        if i > 0 and slowmo:
            prev_v = trajectory[i-1][1]
            interpolated = _interpolate(prev_v, v, n_frames=8)
            frames.extend(interpolated)
        else:
            frames.append(v)
    
    # Add the start frame
    frames = [trajectory[0][1]] + frames
    
    # Build the animation
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Bloch sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v_sphere = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v_sphere))
    y_sphere = np.outer(np.sin(u), np.sin(v_sphere))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v_sphere))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.1)
    
    # Axes
    ax.plot([0, 1.2], [0, 0], [0, 0], 'r-', alpha=0.5)
    ax.plot([0, 0], [0, 1.2], [0, 0], 'g-', alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, 1.2], 'b-', alpha=0.5)
    ax.text(1.2, 0, 0, 'X', color='red')
    ax.text(0, 1.2, 0, 'Y', color='green')
    ax.text(0, 0, 1.2, 'Z', color='blue')
    
    # Trail (fading)
    trail = ax.plot([], [], [], 'o-', color='orange', alpha=0.6, markersize=3, linewidth=1)
    # Current vector (line from origin to vector tip)
    arrow, = ax.plot([0, 0], [0, 0], [0, 0], color='red', linewidth=2)   
    # Label
    text = ax.text(0, 0, 1.3, '', ha='center', fontsize=9)
    
    path_so_far = []
    
    def animate(frame_idx):
        v = frames[frame_idx]
        path_so_far.append(v)
        
        # Update trail
        if len(path_so_far) > 1:
            trail_x = [p[0] for p in path_so_far]
            trail_y = [p[1] for p in path_so_far]
            trail_z = [p[2] for p in path_so_far]
            trail[0].set_data(trail_x, trail_y)
            trail[0].set_3d_properties(trail_z)
        
        # Update arrow
        arrow.set_data([0, v[0]], [0, v[1]])
        arrow.set_3d_properties([0, v[2]])   
        
        # Update label
        theta = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
        phi = np.degrees(np.arctan2(v[1], v[0]))
        p0 = (1 + v[2]) / 2
        text.set_text(f"θ={theta:.0f}° φ={phi:.0f}° P(0)={p0:.2f}")
        
        ax.view_init(elev=20, azim=45)
        return []
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    
    # Build trajectory JSON (for replay)
    replay_data = []
    for label, v in trajectory:
        theta = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
        phi = np.degrees(np.arctan2(v[1], v[0]))
        p0 = (1 + v[2]) / 2
        replay_data.append({
            "gate": label,
            "theta_deg": round(float(theta), 2),
            "phi_deg": round(float(phi), 2),
            "vector": [round(float(x), 6) for x in v],
            "P_0": round(float(p0), 6),
            "P_1": round(float(1 - p0), 6),   
        })
    
    return {
        "gif_path": output_path,
        "frames": len(frames),
        "slowmo": slowmo,
        "trajectory": replay_data,
    }


if __name__ == "__main__":
    print("=== CUDANT_VQPC: VISUALIZE (DEMO) ===\n")
    
    # Demo 1: Clean circuit (straight path)
    print("Generating: Rx(50) — clean path...")
    r = generate_gif("Rx(50)", slowmo=True, output_path="replay_clean.gif")
    print(f"  {r['gif_path']}: {r['frames']} frames\n")
    
    # Demo 2: Wobbly circuit (cone)
    print("Generating: 5x[Rx(5), Rz(10)] — wobbly path...")
    gates_wobbly = ", ".join(["Rx(5), Rz(10)"] * 5)
    r = generate_gif(gates_wobbly, slowmo=True, output_path="replay_wobbly.gif")
    print(f"  {r['gif_path']}: {r['frames']} frames\n")
    
    # Demo 3: Phase kickback (H, T, H)
    print("Generating: H, T, H — phase kickback...")
    r = generate_gif("H, T, H", slowmo=True, output_path="replay_kickback.gif")
    print(f"  {r['gif_path']}: {r['frames']} frames\n")
    
    print("=== VISUALIZE BUILT ===")
    print("  3 GIFs in your project folder.")
    print("  Open them in any image viewer.")
    print("  The red arrow is the particle. The orange trail is its path.")
    print("  Clean = straight line. Wobbly = cone. Kickback = curve.")   
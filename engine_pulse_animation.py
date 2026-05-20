import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi

# =====================================================================
# CORE ARCHITECTURE & ABSOLUTE STATOR LAYER (Operationalized)
# =====================================================================
class Ternary369Layer(nn.Module):
    def __init__(self, in_features=8, hidden_nodes=9, coupling_strength=1.3, anchor_strength=2.5):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.coupling_strength = coupling_strength
        self.anchor_strength = anchor_strength
        
        self.input_proj = nn.Linear(in_features, hidden_nodes)
        self.coupling = nn.Parameter(torch.randn(hidden_nodes, hidden_nodes) * 0.04)
        
        self.idx3, self.idx6, self.idx9 = 2, 5, 8

    def forward_with_tracking(self, x):
        """Returns both the output and the raw hidden activations for animation"""
        h = self.input_proj(x)
        
        # 1. 3-6 Kinetic Cross-Coupling
        coupled = torch.matmul(h, self.coupling)
        for b in range(h.shape[0]):
            diff = h[b, self.idx3] - h[b, self.idx6]
            coupled[b, self.idx3] += self.coupling_strength * torch.sin(diff)
            coupled[b, self.idx6] += self.coupling_strength * torch.sin(-diff)
            for i in range(self.hidden_nodes):
                if i not in (self.idx3, self.idx6, self.idx9):
                    coupled[b, i] += 0.35 * self.coupling_strength * (h[b, self.idx3] - h[b, i])
        h = h + 0.65 * coupled
        
        # 2. Absolute Stator Projection
        mean = h.mean(dim=1, keepdim=True)
        h = h - self.anchor_strength * mean
        h[:, self.idx9] = 0.0  # Absolute zero anchor
        h = h * 0.82
        
        return h

# =====================================================================
# GENERATING THE EXPONENTIAL DRIFT LOAD
# =====================================================================
seq_len = 300
t = np.arange(seq_len, dtype=np.float32)
base = np.sin(t * 0.07) + 0.5 * np.sin(t * 0.2)
drift = np.exp(t * 0.004) - 1.0  # Aggressive exponential acceleration

data_np = np.zeros((seq_len, 8), dtype=np.float32)
for i in range(8):
    data_np[:, i] = base + drift + i * 0.18 + np.random.randn(seq_len) * 0.05
data_tensor = torch.tensor(data_np)

# Initialize Layer
layer = Ternary369Layer()
with torch.no_grad():
    hidden_history = layer.forward_with_tracking(data_tensor).numpy()

# =====================================================================
# GEOMETRIC CIRCLE ROUTING & ANIMATION ENGINE
# =====================================================================
angles = [i * (2 * pi / 9) + (pi / 2) for i in range(9)]
angles = [a if a >= 0 else a + 2*pi for a in angles]
x_coords = np.cos(angles)
y_coords = np.sin(angles)

fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

# Establish visual network layout
lines = []
for i in range(9):
    for j in range(i+1, 9):
        # Emphasize the 3-6 kinetic bridge link in gold, others in muted blue
        if (i == 2 and j == 5) or (i == 5 and j == 2):
            line, = ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                            color='#ffd700', alpha=0.6, linewidth=3, zorder=1)
        else:
            line, = ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                            color='#1f6feb', alpha=0.15, linewidth=1, zorder=1)
        lines.append(line)

# Plot nodes
scatter = ax.scatter(x_coords, y_coords, c=np.zeros(9), cmap='plasma', 
                     s=450, edgecolors='white', linewidths=1.5, zorder=2, vmin=-2, vmax=2)

# Labeling nodes matching the manual's structural positions
labels = ['1', '2', '3\n(Kinetic)', '4', '5', '6\n(Kinetic)', '7', '8', '9\n(Stator)']
for i, txt in enumerate(labels):
    offset_x, offset_y = x_coords[i] * 1.22, y_coords[i] * 1.22
    color = '#ffd700' if i in [2, 5] else ('#ff4500' if i == 8 else '#c9d1d9')
    ax.text(offset_x, offset_y, txt, ha='center', va='center', 
            color=color, fontsize=11, fontweight='bold')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')
title = ax.text(0, 1.4, "", ha='center', va='center', color='#c9d1d9', fontsize=12, fontweight='bold')

def update(frame):
    states = hidden_history[frame]
    scatter.set_array(states)
    
    # Dynamically scale sizes based on kinetic energy expression
    sizes = 450 + np.abs(states) * 300
    scatter.set_sizes(sizes)
    
    # Highlight lines connecting to active kinetic elements
    intensity = min(1.0, max(0.1, np.abs(states[2] - states[5]) * 0.5))
    lines[13].set_alpha(intensity)  # Explicitly tracks the 3-6 index line
    
    title.set_text(f"Spantelergia Engine Processing\nTime Step: {frame} | Exponential Drift Profile")
    return [scatter, title] + lines

ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=40, blit=True)
plt.show()

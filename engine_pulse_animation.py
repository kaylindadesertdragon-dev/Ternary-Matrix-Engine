import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi
import time

# =====================================================================
# CORE TERNARY 369 LAYER (unchanged core logic)
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
        h = self.input_proj(x)
        
        coupled = torch.matmul(h, self.coupling)
        for b in range(h.shape[0]):
            diff = h[b, self.idx3] - h[b, self.idx6]
            coupled[b, self.idx3] += self.coupling_strength * torch.sin(diff)
            coupled[b, self.idx6] += self.coupling_strength * torch.sin(-diff)
            for i in range(self.hidden_nodes):
                if i not in (self.idx3, self.idx6, self.idx9):
                    coupled[b, i] += 0.35 * self.coupling_strength * (h[b, self.idx3] - h[b, i])
        h = h + 0.65 * coupled
        
        mean = h.mean(dim=1, keepdim=True)
        h = h - self.anchor_strength * mean
        h[:, self.idx9] = 0.0
        h = h * 0.82
        
        return h

# =====================================================================
# ADVANCED WAVE PHYSICS VISUALIZER
# =====================================================================
class WavePhysics369Visualizer:
    def __init__(self):
        self.layer = Ternary369Layer()
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='#0d1117')
        self.ax.set_facecolor('#0d1117')
        
        # Node geometry
        self.angles = np.array([i * (2 * pi / 9) + (pi / 2) for i in range(9)])
        self.x_coords = np.cos(self.angles)
        self.y_coords = np.sin(self.angles)
        self.node_pos = np.stack((self.x_coords, self.y_coords), axis=1)
        
        # Background mesh for interference field
        self.res = 120
        x = np.linspace(-1.55, 1.55, self.res)
        y = np.linspace(-1.55, 1.55, self.res)
        self.X, self.Y = np.meshgrid(x, y)
        self.field = np.zeros((self.res, self.res))
        
        self.field_img = self.ax.imshow(self.field, extent=[-1.55, 1.55, -1.55, 1.55],
                                        cmap='plasma', alpha=0.45, zorder=0, vmin=-2, vmax=2)
        
        # Network lines
        self.lines = []
        for i in range(9):
            for j in range(i + 1, 9):
                color = '#ffd700' if {i, j} == {2, 5} else '#1f6feb'
                alpha = 0.75 if {i, j} == {2, 5} else 0.10
                lw = 4 if {i, j} == {2, 5} else 1
                line, = self.ax.plot([self.x_coords[i], self.x_coords[j]],
                                   [self.y_coords[i], self.y_coords[j]],
                                   color=color, alpha=alpha, linewidth=lw, zorder=1)
                self.lines.append(line)
        
        # Nodes
        self.scatter = self.ax.scatter(self.x_coords, self.y_coords, c=np.zeros(9),
                                       cmap='plasma', s=520, edgecolors='white',
                                       linewidths=2.5, zorder=4, vmin=-3.5, vmax=3.5)
        
        # Labels
        labels = ['1', '2', '3\n(Wave Source)', '4', '5', '6\n(Wave Source)', '7', '8', '9\n(Stator)']
        for i, txt in enumerate(labels):
            col = '#ffd700' if i in [2, 5] else ('#ff4500' if i == 8 else '#c9d1d9')
            self.ax.text(self.x_coords[i]*1.28, self.y_coords[i]*1.28, txt,
                        ha='center', va='center', color=col, fontsize=11, fontweight='bold')
        
        self.ax.set_xlim(-1.65, 1.65)
        self.ax.set_ylim(-1.65, 1.65)
        self.ax.axis('off')
        
        self.title = self.ax.text(0, 1.52, "Ternary 369 • Phase Field Interference Engine",
                                  ha='center', va='center', color='#c9d1d9', fontsize=14, fontweight='bold')
        
        self.time_step = 0
        self.phase3 = 0.0
        self.phase6 = 0.0
    
    def generate_input(self):
        t = self.time_step * 0.042
        base = np.sin(t * 1.1) + 0.7 * np.sin(t * 2.8)
        drift = np.exp(self.time_step * 0.0042) - 1.0
        noise = np.random.randn(8) * 0.055
        x = np.full(8, base + drift) + np.arange(8) * 0.185 + noise
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    
    def compute_interference_field(self, states):
        """Radiating interference from nodes 3 and 6"""
        field = np.zeros((self.res, self.res))
        
        pos3 = self.node_pos[2]
        pos6 = self.node_pos[5]
        
        self.phase3 += 0.12 + states[2] * 0.08
        self.phase6 += 0.12 + states[5] * 0.08
        delta_phase = self.phase3 - self.phase6
        
        for i in range(self.res):
            for j in range(self.res):
                p = np.array([self.X[i,j], self.Y[i,j]])
                
                r3 = np.linalg.norm(p - pos3) + 1e-6
                r6 = np.linalg.norm(p - pos6) + 1e-6
                
                wave3 = np.sin(self.phase3 - r3 * 9.5) / (r3 ** 0.6)
                wave6 = np.sin(self.phase6 - r6 * 9.5) / (r6 ** 0.6)
                
                interference = wave3 + wave6 + 0.8 * np.sin(delta_phase + (r3 + r6) * 4.2)
                field[i, j] = interference * (1.0 + 0.4 * states[2] + 0.4 * states[5])
        
        return field
    
    def update(self, frame):
        x = self.generate_input()
        with torch.no_grad():
            states = self.layer.forward_with_tracking(x).squeeze(0).numpy()
        
        self.field = self.compute_interference_field(states)
        self.field_img.set_array(self.field)
        
        pulse = 1.0 + 0.6 * np.abs(states) * (1.0 + np.sin(self.time_step * 0.35))
        modulated = states + 1.2 * np.sin(self.time_step * 0.22 + self.angles * 3.0)
        
        self.scatter.set_array(modulated)
        self.scatter.set_sizes(520 + np.abs(modulated) * 480 * pulse)
        
        kinetic = abs(states[2] - states[5])
        self.lines[13].set_alpha(min(1.0, 0.3 + kinetic * 1.1))
        self.lines[13].set_linewidth(4.0 + kinetic * 5.5)
        
        self.title.set_text(f"369 Phase Field Engine • Step {self.time_step}\n"
                           f"Drift: {np.exp(self.time_step*0.0042)-1.0:.2f} | "
                           f"Δφ: {self.phase3-self.phase6:.2f} | Kinetic Δ: {states[2]-states[5]:.3f}")
        
        self.time_step += 1
        return [self.field_img, self.scatter, self.title] + self.lines

if __name__ == "__main__":
    print("🌊 Launching Advanced Phase Field + Radial Pulsing Engine...")
    viz = WavePhysics369Visualizer()
    ani = animation.FuncAnimation(viz.fig, viz.update, frames=None,
                                  interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
        

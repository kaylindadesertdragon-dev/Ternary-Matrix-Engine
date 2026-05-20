import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SYSTEM PARAMETERS & INITIALIZATION
# ==========================================
N = 9         # Number of nodes representing the ternary framework
T = 200       # Time steps for the simulation
dt = 0.05     # Time increment

def spantelergia_transduction(state, input_vec, anchor=True):
    # Kinetic transduction phase through wave-like 3-6 couplings
    new_state = state + dt * input_vec
    
    if anchor:
        # The 9-Vector Stator Apex acting as an absolute zero-point anchor
        # Pulls the global modular invariant back to baseline equilibrium
        global_sum = np.sum(new_state)
        modular_correction = (global_sum % 9) / N
        new_state = new_state - modular_correction
        
    # Prevent numerical explosion (Standard clipping safeguard)
    new_state = np.clip(new_state, -10, 10)
    return new_state

# ==========================================
# SIMULATION ENGINE
# ==========================================
def run_simulation(with_anchor=True):
    # Establish tiny initial environmental noise
    state = np.zeros(N) + 0.1 * np.random.randn(N)  
    states_history = [state.copy()]
    
    for t in range(T):
        t_norm = t * dt
        # Real-time present-moment kinetic input vector (sinusoidal + noise)
        input_vec = 0.5 * np.sin(2 * np.pi * t_norm / 3 + np.arange(N)) + 0.1 * np.random.randn(N)
        
        state = spantelergia_transduction(state, input_vec, anchor=with_anchor)
        states_history.append(state.copy())
    
    return np.array(states_history)

# Execute both framework states
np.random.seed(42)
history_anchored = run_simulation(with_anchor=True)
history_drifting = run_simulation(with_anchor=False)

# ==========================================
# PLOTTING & VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Stator Anchor active
plt.subplot(1, 2, 1)
plt.plot(history_anchored)
plt.title("With 9-Vector Stator Anchor\n(Stable resonance, low drift)")
plt.xlabel("Time steps")
plt.ylabel("Node States")
plt.grid(True)

# Plot 2: Fibonacci recursive drift active
plt.subplot(1, 2, 2)
plt.plot(history_drifting)
plt.title("Without Anchor\n(Fibonacci-like recursive drift)")
plt.xlabel("Time steps")
plt.ylabel("Node States")
plt.grid(True)

plt.tight_layout()
plt.show()

# Quantify systemic variance data
print("Anchored global sum variance:", np.var(np.sum(history_anchored, axis=1)))
print("Drifting global sum variance:", np.var(np.sum(history_drifting, axis=1)))

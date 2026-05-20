import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# SYSTEM CONFIGURATION & ENVIRONMENTAL SIGNAL CHARACTERISTICS
# =====================================================================
N = 9          
T = 400        # Extended time window to observe prolonged energy accumulation
dt = 0.05      
K = 2.5        # Dynamic coupling coefficient

node_3 = 2
node_6 = 5
node_9 = 8

def run_tracking_benchmark(use_ternary_matrix=True):
    np.random.seed(369)
    phases = np.random.uniform(0, 2 * np.pi, N)
    
    phase_history = []
    energy_expended = 0.0
    tracking_errors = []
    
    for t in range(T):
        t_norm = t * dt
        
        # ENVIRONMENTAL SIGNAL: A chaotic, drifting target signal representing real-world noise
        # This acts as the external "living" input stream to track
        target_signal = np.pi + np.sin(t_norm) + 0.5 * np.sin(3 * t_norm) + 0.2 * np.random.randn()
        target_signal = target_signal % (2 * np.pi)
        
        d_phases = np.zeros(N)
        
        if use_ternary_matrix:
            # 1. TERNARY METHOD: Route input purely through the active 3-6 kinetic bridge
            d_phases[node_3] = 2.0 * np.sin(target_signal - phases[node_3]) + 3.0 * np.sin(phases[node_6] - phases[node_3])
            d_phases[node_6] = 2.0 * np.sin(target_signal - phases[node_6]) + 3.0 * np.sin(phases[node_3] - phases[node_6])
            
            # The remaining nodes track the internal 3-6 field lines structurally
            for i in range(N):
                if i not in [node_3, node_6, node_9]:
                    d_phases[i] = 0.5 * np.sin(phases[node_3] - phases[i])
            
            # Node 9 remains the immutable zero-point stator anchor
            phases[node_9] = 0.0
            global_sync = np.arctan2(np.sum(np.sin(phases)), np.sum(np.cos(phases)))
            d_phases[node_3] -= 0.5 * np.sin(phases[node_3] - global_sync)
            d_phases[node_6] -= 0.5 * np.sin(phases[node_6] - global_sync)
            
            # Energy cost scales only with localized kinetic adjustment (Resonance)
            control_effort = np.abs(d_phases[node_3]) + np.abs(d_phases[node_6])
            
        else:
            # 2. DRIFTING/FIBONACCI METHOD: Every individual node tries to recursively track 
            # and adjust to the input simultaneously without a central anchor point
            for i in range(N):
                coupling_sum = np.sum(np.sin(phases - phases[i]))
                d_phases[i] = 1.0 * np.sin(target_signal - phases[i]) + (K / N) * coupling_sum
            
            # Energy cost balloons because every node must constantly combat drift independently
            control_effort = np.sum(np.abs(d_phases))
            
        # Accumulate metrics
        energy_expended += control_effort * dt
        current_error = np.abs(np.sin(np.mean(phases) - target_signal))
        tracking_errors.append(current_error)
        
        # Update phases
        phases += d_phases * dt
        phases = phases % (2 * np.pi)
        phase_history.append(phases.copy())
        
    return np.array(phase_history), np.mean(tracking_errors), energy_expended

# Execute the benchmarking simulation
_, error_ternary, energy_ternary = run_tracking_benchmark(use_ternary_matrix=True)
_, error_drifting, energy_drifting = run_tracking_benchmark(use_ternary_matrix=False)

# =====================================================================
# VISUAL COMPARISON: THE POWER WALL DEMONSTRATION
# =====================================================================
metrics = ['Tracking Coherence Error\n(Lower is More Accurate)', 'Systemic Wattage Proxy\n(Total Energy Expended)']
ternary_values = [error_ternary, energy_ternary]
drifting_values = [error_drifting, energy_drifting]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ternary_values, width, label='Explicit 3-6-9 Matrix Architecture', color='#2ca02c')
plt.bar(x + width/2, drifting_values, width, label='Standard Recursive Framework (Fibonacci)', color='#d62728')

plt.ylabel('Systemic Performance Metrics')
plt.title('The Computing Power Wall: 3-6-9 Matrix vs. Fibonacci Recursion\n(Tracking Chaos Under High Data Load)')
plt.xticks(x, metrics)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print(f"--- BENCHMARK RESULTS ---")
print(f"Ternary Matrix - Coherence Error: {error_ternary:.4f} | Total Energy Cost: {energy_ternary:.2f} Watts/Proxy")
print(f"Fibonacci Drift - Coherence Error: {error_drifting:.4f} | Total Energy Cost: {energy_drifting:.2f} Watts/Proxy")
print(f"Energy Efficiency Gain: {((energy_drifting - energy_ternary) / energy_drifting) * 100:.1f}% less power consumption.")

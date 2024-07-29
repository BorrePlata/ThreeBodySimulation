import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
MASS_EARTH = 5.972e24  # Mass of the Earth (kg)
MASS_MOON = 7.348e22  # Mass of the Moon (kg)
MASS_SUN = 1.989e30  # Mass of the Sun (kg)

# Initial conditions for the three bodies (positions in meters and velocities in meters/second)
INITIAL_CONDITIONS = np.array([
    [1.496e11, 0, 0, 29.78e3],  # Earth
    [1.496e11 + 3.844e8, 0, 0, 29.78e3 + 1.022e3],  # Moon
    [0, 0, 0, 0]  # Sun
]).flatten()

# Time span for the simulation (seconds)
TIME_SPAN = (0, 30 * 24 * 3600)  # 30 days
TIME_EVAL = np.linspace(*TIME_SPAN, 10000)  # Higher resolution

# Output directory
OUTPUT_DIR = "simulation_results"

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def three_body_equations(t, y, m1, m2, m3):
    r1, r2, r3 = y[:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    r12 = np.linalg.norm(r1 - r2)
    r13 = np.linalg.norm(r1 - r3)
    r23 = np.linalg.norm(r2 - r3)

    a1 = -G * m2 * (r1 - r2) / r12**3 - G * m3 * (r1 - r3) / r13**3
    a2 = -G * m1 * (r2 - r1) / r12**3 - G * m3 * (r2 - r3) / r23**3
    a3 = -G * m1 * (r3 - r1) / r13**3 - G * m2 * (r3 - r2) / r23**3

    return np.concatenate((v1, v2, v3, a1, a2, a3))

def run_simulation(initial_conditions, t_span, t_eval, masses):
    # Use tqdm to add a progress bar
    with tqdm(total=len(t_eval), desc="Simulating", unit="step") as pbar:
        def progress_func(t, y):
            pbar.update()
            return three_body_equations(t, y, *masses)

        sol = solve_ivp(progress_func, t_span, initial_conditions, t_eval=t_eval, rtol=1e-12, atol=1e-15, method='DOP853')
    return sol

def save_simulation_data(sol, filename):
    df = pd.DataFrame({
        'time': sol.t,
        'r1_x': sol.y[0], 'r1_y': sol.y[1],
        'r2_x': sol.y[2], 'r2_y': sol.y[3],
        'r3_x': sol.y[4], 'r3_y': sol.y[5]
    })
    df.to_csv(filename, index=False)

def plot_trajectories(sol, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(sol.y[0], sol.y[1], label='Body 1 (Earth)')
    plt.plot(sol.y[2], sol.y[3], label='Body 2 (Moon)')
    plt.plot(sol.y[4], sol.y[5], label='Body 3 (Sun)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Three-Body Problem Simulation')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def analyze_energy(sol, masses):
    m1, m2, m3 = masses
    r1, r2, r3 = sol.y[:2], sol.y[2:4], sol.y[4:6]
    v1, v2, v3 = sol.y[6:8], sol.y[8:10], sol.y[10:12]
    
    kinetic_energy = 0.5 * (m1 * np.sum(v1**2, axis=0) + m2 * np.sum(v2**2, axis=0) + m3 * np.sum(v3**2, axis=0))
    potential_energy = (
        -G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0) 
        -G * m1 * m3 / np.linalg.norm(r1 - r3, axis=0) 
        -G * m2 * m3 / np.linalg.norm(r2 - r3, axis=0)
    )
    total_energy = kinetic_energy + potential_energy
    
    return kinetic_energy, potential_energy, total_energy

def plot_energy(time, kinetic, potential, total, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(time, kinetic, label='Kinetic Energy')
    plt.plot(time, potential, label='Potential Energy')
    plt.plot(time, total, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    ensure_directory_exists(OUTPUT_DIR)
    
    masses = (MASS_EARTH, MASS_MOON, MASS_SUN)
    sol = run_simulation(INITIAL_CONDITIONS, TIME_SPAN, TIME_EVAL, masses)

    # Save simulation data
    data_filename = os.path.join(OUTPUT_DIR, "three_body_simulation.csv")
    save_simulation_data(sol, data_filename)

    # Save plot
    plot_filename = os.path.join(OUTPUT_DIR, "three_body_trajectories.png")
    plot_trajectories(sol, plot_filename)

    # Energy analysis
    kinetic_energy, potential_energy, total_energy = analyze_energy(sol, masses)
    energy_filename = os.path.join(OUTPUT_DIR, "energy_analysis.png")
    plot_energy(sol.t, kinetic_energy, potential_energy, total_energy, energy_filename)

    # Save metadata
    metadata = {
        "initial_conditions": INITIAL_CONDITIONS.tolist(),
        "time_span": TIME_SPAN,
        "masses": {
            "earth": MASS_EARTH,
            "moon": MASS_MOON,
            "sun": MASS_SUN
        },
        "data_filename": data_filename,
        "plot_filename": plot_filename,
        "energy_filename": energy_filename
    }
    metadata_filename = os.path.join(OUTPUT_DIR, "simulation_metadata.json")
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()

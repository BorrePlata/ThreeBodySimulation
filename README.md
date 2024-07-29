# ThreeBodySimulation

A detailed simulation of the three-body problem in celestial mechanics using Python. This project includes numerical solutions to the three-body problem, trajectory visualizations, and energy analysis.

## Features

- Numerical integration of the three-body problem using `scipy.integrate.solve_ivp`
- Real-time simulation progress using `tqdm`
- Detailed trajectory plots for each body
- Comprehensive energy analysis (kinetic, potential, and total energy)
- High precision and stability using advanced numerical methods

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BorrePlata/ThreeBodySimulation.git
cd ThreeBodySimulation
pip install -r requirements.txt
Usage
Run the simulation:

bash
Copiar código
python main.py
The simulation results will be saved in the simulation_results directory, including:

three_body_simulation.csv: CSV file containing the simulation data
three_body_trajectories.png: Plot of the trajectories of the three bodies
energy_analysis.png: Plot of the energy analysis over time
simulation_metadata.json: Metadata for the simulation
Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

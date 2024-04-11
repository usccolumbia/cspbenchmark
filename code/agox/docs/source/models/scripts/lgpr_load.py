import numpy as np
import matplotlib.pyplot as plt

from agox.models import load
from agox.models.datasets import datasets

data = datasets['Ag5O3']
test_data = data[80:]

model = load('my-model.pkl')

true_energies = np.array([d.get_potential_energy() for d in test_data])
pred_energies = model.predict_energies(test_data)

# Make parity plot
fig, ax = plt.subplots()
ax.scatter(true_energies, pred_energies)

min_energy, max_energy, q = np.min(np.vstack((true_energies, pred_energies))), np.max(np.vstack((true_energies, pred_energies))), 0.2
ax.plot([min_energy-q, max_energy+q], [min_energy-q, max_energy+q], '-k')
ax.set_xlim(([min_energy-q, max_energy+q]))
ax.set_ylim(([min_energy-q, max_energy+q]))
ax.set_xlabel('True Energies')
ax.set_ylabel('Predicted Energies')
ax.set_title(f'Test MAE: {np.mean(np.abs(true_energies-pred_energies)/len(data[0]))*1000:.2f} meV/atom')
plt.savefig('parity.png')

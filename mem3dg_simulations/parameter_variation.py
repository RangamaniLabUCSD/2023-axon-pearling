import numpy as np
import pint

osmolarities = np.linspace(100, 800, 8) / 1000
bending_moduli = np.linspace(20, 100, 9)
tensions = np.logspace(-3, -1, 3)

target_volume_scale = 3
reservoir_volume=700

# print(osmolarities, bending_moduli, tensions)

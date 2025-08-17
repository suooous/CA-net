# %%
import numpy as np

weight = np.array([0.2, 0.4, 0.4])
scores = np.array([86.65, 87, 86])
final = (weight * scores).sum()
print(final)

# %%
weight = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15])
scores = np.array([85, 84, 84, 85, 82, 86, 82])
final = (weight * scores).sum()
print(final)
# %%





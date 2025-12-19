# Utilities: Data Loaders and Visualization

import matplotlib.pyplot as plt
import pandas as pd

def plot_paths(paths, title="Simulated Paths"):
    plt.figure(figsize=(10, 6))
    plt.plot(paths.T)
    plt.title(title)
    plt.show()

def load_vix_data(filepath):
    return pd.read_csv(filepath)

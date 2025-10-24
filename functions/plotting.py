import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
from scipy.interpolate import griddata

def single_gaze_plot(self, ax, metric, method):
    if metric == 'Total Cost':
        cost = self.cost_df[('total_cost', 'cost')]
    elif metric == 'Volume Penalty':
        cost = self.cost_df[('dvh_points', 'dose_volume_penalty')]
    else:
        cost = self.cost_df[('dvh_points', metric)]
    
    

    polar = self.cost_df[('gaze_angles', 'polar')]
    azimuthal = self.cost_df[('gaze_angles', 'azimuthal')]
    polar_opt_metric, azimuthal_opt_metric, cost_opt_metric = self.find_optimum_for_metric(metric)

    if method == 'contourf':
        levels=30
        theta = np.deg2rad(azimuthal)
        r_lin=np.linspace(polar.min(), polar.max(), 100)
        theta_lin = np.linspace(0, 2*np.pi, 100)
        R_grid, Theta_grid = np.meshgrid(r_lin, theta_lin, indexing='ij')
        points = np.column_stack((polar, theta))
        Z_grid = griddata(points, cost, (R_grid, Theta_grid), method='cubic')
        sc = ax.contourf(Theta_grid, R_grid, Z_grid, levels=30, cmap='viridis')

    elif method == 'scatter':
        sc = ax.scatter(np.deg2rad(azimuthal), polar, c=cost, cmap='viridis', s=60)

    opt_metric = ax.scatter(np.deg2rad(azimuthal_opt_metric), polar_opt_metric, marker='*', color='red', s=150)
    if metric in ['Total Cost', 'Volume Penalty']:
        plt.colorbar(sc, ax=ax, fraction=0.05, label='Cost')
    else:
        plt.colorbar(sc, ax=ax, fraction=0.05, label='cGy')
    
    ax.set_title(metric)


def full_plot(self, method='scatter'):
    fig, axes = plt.subplots(2, 4, figsize=(12,8), subplot_kw={'projection': 'polar'}, layout='constrained')
    if self.use_precalculated:
        path = self.extensive_path
    else: path = self.save_path

    with open(path, mode="r", newline="", encoding="utf-8") as file:
        self.cost_df = pd.read_csv(file, header = [0,1])

    self.single_gaze_plot(ax=axes.flat[0], metric='Total Cost', method=method)
    self.single_gaze_plot(ax=axes.flat[1], metric='Volume Penalty', method=method)

    for metric, ax in zip(self.weights, axes.flat[2:]):
        self.single_gaze_plot(ax=ax, metric=metric, method=method)

    plt.suptitle(f'Patient {self.patient_id}')
    plt.savefig(f'plots/{self.patient_id}_scatter_metrics.png', dpi=200)


def find_optimum_for_metric(self, metric):
    """"
    Returns polar, azimuthal and cost with lowest value for metric given
    """
    if metric == 'Total Cost':
        idx_min = self.cost_df[('total_cost', 'cost')].idxmin()
        cost = self.cost_df.loc[idx_min, ('total_cost', 'cost')]
    
    elif metric == 'Volume Penalty':
        idx_min = self.cost_df[('dvh_points', 'dose_volume_penalty')].idxmin()
        cost = self.cost_df.loc[idx_min, ('dvh_points', 'dose_volume_penalty')]

    else:
        idx_min = self.cost_df[('dvh_points', metric)].idxmin()    
        cost = self.cost_df.loc[idx_min, ('dvh_points', metric)]
    
    polar = self.cost_df.loc[idx_min, ('gaze_angles', 'polar')]
    azimuthal = self.cost_df.loc[idx_min, ('gaze_angles', 'azimuthal')]

    return polar, azimuthal, cost

def find_optimum_cost(self):
    """
    Returns polar, azimuthal and cost with lowest total cost
    """
    idx_min = self.cost_df[('total_cost', 'cost')].idxmin()
    polar = self.cost_df.loc[idx_min, ('gaze_angles', 'polar')]
    azimuthal = self.cost_df.loc[idx_min, ('gaze_angles', 'azimuthal')]
    cost = self.cost_df.loc[idx_min, ('total_cost', 'cost')]
    return polar, azimuthal, cost

def find_all_optima(self):
    """Returns dictionary {'metric': (polar, azimuthal, cost), ...} 
    containing angles and cost of optimum for each metric
    """
    polar_cost, azimuthal_cost, total_cost = self.find_optimum_cost() 
    optima_dict = {('total_cost'): (polar_cost, azimuthal_cost, total_cost)}

    for metric in self.weights:
        optima_dict[metric] = self.find_optimum_for_metric(metric)
    
    return optima_dict

def scatter_only_optima(self, ax):
    """
    Plot only maxima of each
    """
    optima = self.find_all_optima()
    marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'X', 'P', '*', 'H'])
    for metric, marker in zip(optima, marker_cycle):
        polar, azimuthal, cost = optima[metric]
        sc = ax.scatter(np.deg2rad(azimuthal), polar, s=80, label=metric, marker=marker)
    
    ax.legend(
        title='Type',
        loc='center left',           # place legend to the left/right of bbox
        bbox_to_anchor=(1.1, 0.5),   # (x, y) offset in axes fraction coordinates
        frameon=False
    )




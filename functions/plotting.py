import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import h5py
from scipy.interpolate import griddata


def single_gaze_plot(self, ax, metric):
    #Get data fr
    with h5py.File(self.h5py_file_path, "r") as f:
        gaze_angle_keys = f['gaze_angles'].keys()
        angles = np.array([f['gaze_angles'][gaze_angle_key].attrs['gaze_angle'] for gaze_angle_key in gaze_angle_keys])
        cost = np.array([f['gaze_angles'][gaze_angle_key].attrs[metric] for gaze_angle_key in gaze_angle_keys])

    polar = angles[:,0]
    azimuthal = angles[:,1]
    theta = np.deg2rad(azimuthal)

    polar_opt, azimuthal_opt, _ = self.find_optimum_for_metric('total_cost')
    polar_opt_metric, azimuthal_opt_metric, cost_opt_metric = self.find_optimum_for_metric(metric)
    sc = ax.scatter(theta, polar, c=cost, cmap='viridis', s=60)

    if metric != 'total_cost':
        opt_metric = ax.scatter(np.deg2rad(azimuthal_opt_metric), polar_opt_metric, marker='*', color='gray', s=150, label='Optimum for metric')
    opt = ax.scatter(np.deg2rad(azimuthal_opt), polar_opt, marker='D', color='red', s=100, label='Optimum for total cost')
    if metric in ['total_cost', 'volume_term']:
        label = 'Cost'
    elif 'D' in metric:
        label = 'Gy'
    else:
        label = '% Volume'
        
    plt.colorbar(sc, ax=ax, fraction=0.05, label=label)
    ax.set_title(metric)
    return cost


def full_scatter_plot(self):
    with h5py.File(self.h5py_file_path, "r") as f:
        metrics = list(f['gaze_angles']['0_0'].attrs.keys())
        metrics.remove('gaze_angle')
        n_plots = len(metrics)
        metrics.remove('total_cost')
        metrics.remove('volume_term')


    fig, axes = plt.subplots(2, int(np.ceil(n_plots/2)), figsize=(12,8), subplot_kw={'projection': 'polar'}, layout='constrained')
    self.single_gaze_plot(ax=axes.flat[0], metric='total_cost')
    self.single_gaze_plot(ax=axes.flat[1], metric='volume_term')

    for metric, ax in zip(metrics, axes.flat[2:]):
        self.single_gaze_plot(ax=ax, metric=metric)

    plt.suptitle(f'Patient {self.patient_id}')

    # Get handles and labels from one of the axes (they are the same for all)
    handles, labels = axes.flat[-1].get_legend_handles_labels()

    # Create a single legend for the whole figure
    fig.legend(handles, labels, loc='lower center')
    plt.savefig(f'{self.plot_folder}/scatter_metrics.png', dpi=200)


def dvh_metric_plot(self, metric):
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax_scatter = fig.add_subplot(1, 2, 1, projection='polar')
    ax_lines = fig.add_subplot(1, 2, 2)
    cost = self.single_gaze_plot(ax_scatter, 'total_cost')
    x, roi = metric.split('_')
    cmap = plt.cm.viridis
    colors = cmap((cost - cost.min()) / (cost.max() - cost.min()))
    min_idx = np.argmin(cost)
    y = np.linspace(0, 1, self.num_dvh_bins)
    with h5py.File(self.h5py_file_path, "r") as f:
        gaze_angle_keys = f['gaze_angles'].keys()
        for i, (gaze_angle_key, c) in enumerate(zip(gaze_angle_keys, colors)):
            dvh = np.array(f['gaze_angles'][gaze_angle_key][roi][:])
            if i != min_idx: ax_lines.plot(dvh, y, color=c, zorder=1)
            else: 
                ax_lines.plot(dvh, y, color='red', zorder=1000)
    what = x[0]
    value = int(x[1:])
    if what == 'D':
        ax_lines.axhline(y=value/100, color='gray', linestyle='--', linewidth=1, label=metric)

    elif what == 'V':
        print(value)
        ax_lines.axvline(x=value*100, color='gray', linestyle='--', linewidth=1, label=metric)
    ax_lines.legend(loc='upper right')
    ax_lines.set_title(roi)
    ax_lines.set_xlabel('Dose (cGy)')
    ax_lines.set_ylabel('Rel. Volume')

          
def full_metric_dvh_plot(self):
    for metric in self.weights:
        self.dvh_metric_plot(metric)
        plt.savefig(f'{self.plot_folder}/dvh_{metric}.png', dpi=200)
        print(metric)


def find_optimum_for_metric(self, metric):
    """"
    Returns polar, azimuthal and cost with lowest value for metric given
    """
    with h5py.File(self.h5py_file_path, "r") as f:
        gaze_angle_keys = f['gaze_angles'].keys()
        angles = np.array([f['gaze_angles'][gaze_angle_key].attrs['gaze_angle'] for gaze_angle_key in gaze_angle_keys])
        costs = np.array([f['gaze_angles'][gaze_angle_key].attrs[metric] for gaze_angle_key in gaze_angle_keys])
    
    polar = angles[:,0]
    azimuthal = angles[:,1]
    idx = np.argmin(costs)

    return polar[idx], azimuthal[idx], costs[idx]



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




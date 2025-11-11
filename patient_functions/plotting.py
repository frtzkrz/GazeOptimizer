import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import h5py
from scipy.interpolate import griddata
import os


def single_gaze_plot(self, ax, metric, filtered_gaze_angle_keys=None):
    """
    Add gaze scatter plot to ax.
    ax: plt.axis
    metric: i.e. 'D2_Macula'
    f: h5py file
    """
    vmin = np.min(self.costs)
    vmax = np.max(self.costs)
    #if no gaze angle keys are provided, plot all gaze angles
    if filtered_gaze_angle_keys is None:
        costs = [self.metrics_dict[gaze_angle_key][metric] for gaze_angle_key in self.gaze_angle_keys]
        #for all gaze angles scatter plot with color corresponding to cost of said angle
        sc = ax.scatter(self.theta, self.polar, c=costs, cmap='viridis', s=60)
    else:
        #costs = [self.metrics_dict[gaze_angle_key][metric] for gaze_angle_key in filtered_gaze_angle_keys]
        #for filtered gaze angles scatter plot with color corresponding to cost of said angle
        polars, azimuthals, costs = self.get_gaze_angles_and_costs_from_keys(filtered_gaze_angle_keys=filtered_gaze_angle_keys, metric=metric)
        sc = ax.scatter(np.deg2rad(azimuthals), polars, c=costs, cmap='viridis', s=60)

        #plot new optimum gaze angle if filtering was applied
        new_optimal_gaze_angle_key = self.find_new_optimal_gaze_angles(filtered_gaze_angle_keys)['New Optimum']
        new_optimal_polar, new_optimal_azimuthal = self.gaze_angle_dict[new_optimal_gaze_angle_key]
        ax.scatter(np.deg2rad(new_optimal_azimuthal), new_optimal_polar, c='orange', marker='D', s=60, label='New Optimum after filtering')

    #plot optimum for total cost as red diamond
    opt = ax.scatter(self.optimal_theta, self.optimal_polar, marker='D', color='red', s=50, label='Optimum for total cost')
    
    #we only want to plot metric cost if we have a metric and are not plotting total_cost anyway
    if metric != 'total_cost':
        opt_metric = ax.scatter(self.optimal_theta, self.optimal_polar, marker='*', color='red', s=80, label='Optimum for metric')
    
    #set labels for colorbar
    if metric in ['total_cost', 'volume_term']:
        label = 'Cost'
    elif 'D' in metric:
        label = 'Gy'
    else:
        label = '% Volume'

    plt.colorbar(sc, ax=ax, fraction=0.05, label=label)
    ax.set_title(metric)

def full_scatter_plot(self, plot_folder=None, filtered_gaze_angle_keys=None, filter_dict=None):
    if plot_folder is None:
        plot_folder = f'plots/{self.patient_id}'

    n_plots = len(self.weights) + 2
    
    metrics = list(self.optimal_metrics.keys())

    #create figure
    fig, axes = plt.subplots(2, int(np.ceil(n_plots/2)), figsize=(12,8), subplot_kw={'projection': 'polar'}, layout='constrained')

        #plot scatter for total cost and volume term
    self.single_gaze_plot(ax=axes.flat[0], metric='total_cost', filtered_gaze_angle_keys=filtered_gaze_angle_keys)
    self.single_gaze_plot(ax=axes.flat[1], metric='volume_term', filtered_gaze_angle_keys=filtered_gaze_angle_keys)

    #plot scatter plot for each metric
    for metric, ax in zip(metrics, axes.flat[2:]):
        self.single_gaze_plot(ax=ax, metric=metric, filtered_gaze_angle_keys=filtered_gaze_angle_keys)

    plt.suptitle(f'Patient {self.patient_id}')

    # Get handles and labels from one of the axes (they are the same for all)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    
    # Create a single legend for the whole figure
    fig.legend(handles, labels, loc='lower center')
    if plot_folder is not None: 
        if filtered_gaze_angle_keys is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
    plt.savefig(f'{plot_folder}/scatter_metrics.png', dpi=200)

def plot_gaze_angle_dvhs(self, ax, roi, colors, filtered_gaze_angle_keys):
    y = np.linspace(0, 1, self.num_dvh_bins)

    #for each gaze angle, plot a line 
    for gaze_angle_key, c in zip(self.gaze_angle_keys, colors):

        #get dvh from h5 file
        dvh = self.dvh_dict[roi][gaze_angle_key]

        #plot line of optimal gaze angle in red
        if gaze_angle_key == self.optimal_angle_key:
            ls = '-'
            if filtered_gaze_angle_keys is not None and gaze_angle_key not in filtered_gaze_angle_keys: 
                new_optimal_gaze_angle_key = self.find_new_optimal_gaze_angles(filtered_gaze_angle_keys)['New Optimum']
                new_optimal_dvh = self.dvh_dict[roi][new_optimal_gaze_angle_key]
                ax.plot(new_optimal_dvh, y, color='orange', ls='-', zorder=1000)
            
            ax.plot(dvh, y, color='red', ls=ls, zorder=1000)
            
        #plot lines of non-optimal gaze angles, color according to color in scatter plot
        if filtered_gaze_angle_keys is None or gaze_angle_key in filtered_gaze_angle_keys:
            ax.plot(dvh, y, color=c, zorder=1)
        else: 
            ax.plot(dvh, y, color='gray', zorder=1)

def roi_in_metrics(self, roi):
    #check if there is a metric to this roi
    for metric in self.optimal_metrics:
        if roi in metric:
            return True
    return False

def find_metric_for_roi(self, roi):
    #find metric corresponding to roi
    if not self.roi_in_metrics(roi):
        raise ValueError(f'ROI {roi} not found in metrics.')
    for metric in self.optimal_metrics:
        if roi in metric:
            return metric

def dvh_scatter_plot(self, roi, filtered_gaze_angle_keys=None, ax_lines=None, filter_dict=None):

    #if ax is not given, create new figure
    if ax_lines is None:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)

        #create ax for scatter and dvh
        ax_scatter = fig.add_subplot(1, 2, 1, projection='polar')
        ax_lines = fig.add_subplot(1, 2, 2)

        #plot total cost scatter
        self.single_gaze_plot(ax=ax_scatter, metric='total_cost', filtered_gaze_angle_keys=filtered_gaze_angle_keys)

    #create colormap and scale with min and max of cost
    cmap = plt.cm.viridis
    colors = cmap((self.costs - self.costs.min()) / (self.costs.max() - self.costs.min()))


    
    #plot dvhs for given roi
    self.plot_gaze_angle_dvhs(ax=ax_lines, roi=roi, colors=colors, filtered_gaze_angle_keys=filtered_gaze_angle_keys)

    #if gaze angles were filtered: add marker for cutoff point in dvh
    if filter_dict is not None:
        if filter_dict['roi'] == roi:
            if filter_dict['filter_type']=='D': #D_90 < 50Gy:  'value': 20, 'max': 10, 
                dose = filter_dict['max']*100 #in cGy
                vol = filter_dict['value']/100
            
            else:
                dose = filter_dict['value']*100 #in cGy
                vol = filter_dict['max']/100
            
            #add marker where cut off point is
            ax_lines.plot(dose, vol, 'o', color='orange', markersize=10, label=f'Filter: {filter_dict["filter_type"]}_{filter_dict["value"]} < {filter_dict["max"]}')

    #check if roi has corresponding metric
    is_metric = self.roi_in_metrics(roi)

    #if so, add line to indicate where metric is taken
    if is_metric:
        #find metric corresponding to roi
        metric = self.find_metric_for_roi(roi)

        #find type and value of metric
        x = metric.split('_')[0]
        metric_type = x[0]
        metric_value = int(x[1:])

        #add horizontal/vertical line to indicate where metric is taken
        if metric_type == 'D':
            ax_lines.axhline(y=metric_value/100, color='gray', linestyle='--', linewidth=1, label=metric)

        elif metric_type == 'V':
            ax_lines.axvline(x=metric_value*100, color='gray', linestyle='--', linewidth=1, label=metric)
        
        #finish plot
        ax_lines.legend()
    
    ax_lines.set_title(roi)
    ax_lines.set_xlabel('Dose (cGy)')
    ax_lines.set_ylabel('Rel. Volume')       

def full_metric_dvh_plot(self, plot_folder=None):
    #create figure containing total cost scatter and dvh for each roi

    if plot_folder is None:
        plot_folder = f'plots/{self.patient_id}'

    #for each metric, create separate plot showing scatter of relevant cost on the left and DVHs of corresponding ROI on the right
    for roi in self.rois:
        self.dvh_scatter_plot(roi=roi)
        fig.suptitle(self.patient_id)
        plt.savefig(f'{plot_folder}/dvh_{roi}.png', dpi=200)



def find_contributions(self, gaze_angle_key=None):
    #find contributions of each metric to total cost at given gaze angle
    #returned as dict
    if gaze_angle_key is None:
        gaze_angle_key = self.optimal_angle_key
        
    contributions = {metric: self.weights[metric]*self.metrics_dict[gaze_angle_key][metric] for metric in self.weights}
    contributions['volume_term'] = self.metrics_dict[gaze_angle_key]['volume_term']
    return contributions

def plot_filtered_dvh(self, roi, filtered_gaze_angle_keys):
    #plot dvhs for given roi and filtered gaze angle keys
    y = np.linspace(0, 1, self.num_dvh_bins)
    dvhs = self.dvh_dict[roi]
    plt.figure()

    #plot dvhs for each gaze angle
    for gaze_angle_key in filtered_gaze_angle_keys:
        plt.plot(dvhs[gaze_angle_key], y)
    
def get_gaze_angles_and_costs_from_keys(self, filtered_gaze_angle_keys, metric):
    #return polars, azimuthals, costs for given gaze angle keys and metric
    costs = [self.metrics_dict[k][metric] for k in filtered_gaze_angle_keys if metric in self.metrics_dict[k]]
    polars = [float(key.split('_')[0]) for key in filtered_gaze_angle_keys]
    azimuthals = [float(key.split('_')[1]) for key in filtered_gaze_angle_keys]
    return polars, azimuthals, costs

def full_filtered_metric_dvh_plot(self, filter_dict, filtered_gaze_angle_keys=None, plot_folder=None, save_fig=False):
    #create figure containing total cost scatter and dvh for each roi in a single figure
    #returns filtered gaze angle keys
    
    if plot_folder is None:
        plot_folder = f'plots/{self.patient_id}'

    #filter gaze angles according to filter dict
    if filter_dict['filter_type']=='D':
        filtered_gaze_angle_keys = self.find_gaze_angle_smaller_dvol(roi=filter_dict['roi'], vol=filter_dict['value'], max_dose=filter_dict['max'], filtered_gaze_angle_keys=filtered_gaze_angle_keys)
    
    else:
        filtered_gaze_angle_keys = self.find_gaze_angle_smaller_vdose(roi=filter_dict['roi'], dose=filter_dict['value'], max_vol=filter_dict['max'], filtered_gaze_angle_keys=filtered_gaze_angle_keys)

    fig = plt.figure(figsize=(14,10), constrained_layout=True)

    ax_scatters = fig.add_subplot(3, 4, 1, projection='polar')

    #plot total cost scatter in first subplot
    self.single_gaze_plot(ax=ax_scatters, metric='total_cost', filtered_gaze_angle_keys=filtered_gaze_angle_keys)
    plt.legend()

    #plot dvhs for each roi in remaining subplots
    for i, roi in enumerate(self.rois):
        ax_lines = fig.add_subplot(3, 4, i+2)

        self.dvh_scatter_plot(roi=roi, filtered_gaze_angle_keys=filtered_gaze_angle_keys, ax_lines=ax_lines, filter_dict=filter_dict)

    fig.suptitle(self.patient_id)

    #save figure if desired
    if plot_folder is not None: 
        if filtered_gaze_angle_keys is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/dvhs.png', dpi=200)
    
    
    return filtered_gaze_angle_keys


def compare_contributions_bar(self, angle_key_dict, plot_folder=None, filter_dict=None):
    #compare cost contributions for different gaze angles in bar plot

    #ensure optimal gaze angle is included
    if 'Global Optimum' not in angle_key_dict:
        angle_key_dict['Global Optimum'] = self.optimal_angle_key
    angle_keys = list(angle_key_dict.values())


    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    
    #set width of bars according to number of gaze angles being compared
    width = 0.75/len(angle_key_dict)
    
    #create bars for each gaze angle
    for i, angle_key_name in enumerate(angle_key_dict):
        offset=i*width - width*len(angle_key_dict)/2 + width/2
        contributions = self.find_contributions(gaze_angle_key=angle_key_dict[angle_key_name])
        labels = contributions.keys()
        sizes = [float(contributions[label]) for label in labels]
        x = np.arange(len(labels))  # the label locations
        
        color = 'red' if angle_key_name == 'Global Optimum' else 'orange'
        rects = ax.bar(x + offset, sizes, width, label=angle_key_name, color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Cost Contribution')
    ax.set_title('Cost Contributions by Metric and Gaze Angle')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    #save figure if desired
    if plot_folder is not None: 
        if filter_dict is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/compare_cost_contributions.png', dpi=200)
    

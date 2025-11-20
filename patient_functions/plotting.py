from __future__ import annotations 

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def compare_contributions(
    plans: List[TreatmentPlan], 
    plot_folder: str=None, 
    ax: plt.Axes=None
    ) -> None:

    """
    Compare cost function contributions for different treatment plans.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to compare.
    plot_folder (str, optional): Folder to save the plot. If None, the plot is showed but not saved.
    """
    n_plans = len(plans)
    if n_plans > 4:
        raise ValueError('Too many plans provided. Maximum is 4. Looks messy otherwise')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    
    #set width of bars according to number of gaze angles being compared
    width = 0.75/n_plans
    
    
    #create bars for each gaze angle
    for i, plan in enumerate(plans):
        offset=i*width - width*n_plans/2 + width/2
        contributions = plan.calculate_contributions()
        labels = contributions.keys()
        sizes = [float(contributions[label]) for label in labels]
        x = np.arange(len(labels))  # the label locations
        rects = ax.bar(x + offset, sizes, width, label=plan.name)

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
    else: plt.show()
    
def single_gaze_plot(
    metric: str, 
    plans: List[TreatmentPlan], 
    ax: plt.Axes=None
    ) -> plt.Scatter:

    """
    Create a scatter plot of costs for single gaze angle treatment plans.
    Parameters:
    ax: Matplotlib axis to plot on.
    metric (str): Metric to calculate costs ('total_cost', 'volume_term', or specific DVH metric).
    plans (List[TreatmentPlan]): List of TreatmentPlan objects with single gaze angles.
    Returns:
    sc: Scatter plot object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3), constrained_layout=True)

    for plan in plans:
        if plan.angle_key_2 is not None:
            raise ValueError('single_gaze_plot only works for single gaze angle plans')

    #determine costs based on metric
    if metric == 'total_cost':
        costs = [plan.calculate_cost() for plan in plans]
    
    elif metric == 'volume_term':
        costs = [plan.calculate_volume_term() for plan in plans]

    else: 
        roi = metric.split('_')[1]
        costs = [plan.dvhs[roi].get_metric_value(metric) for plan in plans]

    #extract polar and theta angles from plan angle keys
    polars, thetas = get_angles_from_keys([plan.angle_key for plan in plans], azimuthal_as_radian=True)
    
    sc = ax.scatter(polars, thetas, c=costs, cmap='viridis', s=60)
    return sc

def full_scatter_plot(
    plans: List[TreatmentPlan], 
    plot_folder=None, 
    filter=None
    ) -> None:

    """
    Create a full scatter plot of costs for single gaze angle treatment plans.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects with single gaze angles.
    plot_folder (str, optional): Folder to save the plot. If None, the plot is showed but not saved.
    filter_dict (dict, optional): Dictionary containing filter information for filtered gaze angles.
    """
    
    #get example patient
    pat = plans[0].patient

    #create fig, axes: number of plots = number of weights + 2 (total cost and volume term)
    weights = pat.weights
    n_plots = len(weights) + 2
    fig, axes = plt.subplots(2, int(np.ceil(n_plots/2)), figsize=(12,8), subplot_kw={'projection': 'polar'}, layout='constrained')


    #plot scatter for total cost and volume term
    single_gaze_plot(metric='total_cost', plans=plans, ax=axes.flat[0])
    single_gaze_plot(metric='volume_term', plans=plans, ax=axes.flat[1])

    #plot scatter plot for each metric
    for metric, ax in zip(metrics, axes.flat[2:]):
        single_gaze_plot(metric=metric, plans=plans, ax=ax)

    plt.suptitle(f'Patient {pat.patient_id}')

    # Get handles and labels from one of the axes (they are the same for all)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    
    # Create a single legend for the whole figure
    fig.legend(handles, labels, loc='lower center')

    #save figure if desired
    if plot_folder is not None: 
        if filtered_gaze_angle_keys is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/scatter_metrics.png', dpi=200)
    else: plt.show()



def plot_gaze_angle_dvhs(
    plans: List[TreatmentPlan], 
    roi: str, 
    ax: plt.Axes, 
    colors: List[str], 
    old_plans: List[TreatmentPlan]=None
    ) -> None:

    """
    Plot DVHs for different gaze angles for a specific ROI.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to plot DVHs for.
    roi (str): The region of interest to plot DVHs for.
    ax (plt.Axes): Matplotlib axis to plot on.
    colors (List[str]): List of colors for each gaze angle.
    old_plans (List[TreatmentPlan], optional): Complete list of TreatmentPlan objects if filter was applied.
    """

    best_plan = find_best_plan(plans)

    for plan, color in zip(plans, colors):
        if old_plans is not None:
            #plot filtered out lines in gray
            alpha = 1.
            zorder = 1
            if plan not in old_plans:
                color = 'gray'
                alpha = 0.5
                zorder = 0
        if plan == best_plan:
            color = 'red'
            zorder = 1000

        plan.dvhs[roi].plot(ax=ax, plot_args = {'color': color, 'alpha': alpha, 'zorder': zorder})

    #if old plans provided, plot old best plan's dvh as dashed orange line if different from new best plan
    if old_plans is not None:
        old_best = find_best_plan(old_plans)
        if old_best != best_plan:
            old_best.dvhs[roi].plot(ax=ax, plot_args = {'color': 'orange', 'linestyle': '--', 'zorder': 999})






def draw_metric_line(
    roi: str, 
    weights: Weights, 
    ax: plt.Axes):
    #find type and value of metric
    metric = find_metric_for_roi(roi=roi, weights=weights)
    metric_type = metric.metric.metric_type
    metric_value = metric.metric.value
    #add horizontal/vertical line to indicate where metric is taken
    if metric_type == 'D':
        ax.axhline(y=metric_value/100, color='gray', linestyle='--', linewidth=1, label=metric)

    elif metric_type == 'V':
        ax.axvline(x=metric_value*100, color='gray', linestyle='--', linewidth=1, label=metric)

def dvh_scatter_plot(
    plans: List[TreatmentPlan], 
    roi: str, 
    ax_lines: plt.Axes=None, 
    old_plans: List[TreatmentPlan]=None,
    filter_dict: dict=None,
    plot_folder: str=None
    ) -> None:
    """
    Create a DVH scatter plot for a specific ROI.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to plot DVHs for.
    roi (str): The region of interest to plot DVHs for.
    ax_lines (plt.Axes, optional): Matplotlib axis to plot DVHs on. If None, a new figure is created.
    old_plans (List[TreatmentPlan], optional): Complete list of TreatmentPlan objects if filter was applied.
    filter_dict (dict, optional): Dictionary containing filter information for filtered gaze angles.
    plot_folder (str, optional): Folder to save the plot.
    """

    weights = plans[0].patient.weights
    return_fig = False
    #if ax is not given, create new figure
    if ax_lines is None:
        return_fig = True
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)

        #create ax for scatter and dvh
        ax_scatter = fig.add_subplot(1, 2, 1, projection='polar')
        ax_lines = fig.add_subplot(1, 2, 2)

        #plot total cost scatter
        single_gaze_plot(metric='total_cost', plans=plans, ax=ax_scatter)

    #create colormap and scale with min and max of cost
    costs = np.array([plan.calculate_cost() for plan in plans])
    cmap = plt.cm.viridis
    colors = cmap((costs - costs.min()) / (costs.max() - costs.min()))

    #plot dvhs for given roi
    plot_gaze_angle_dvhs(plans=plans, roi=roi, ax=ax_lines, colors=colors, old_plans=old_plans)

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
    is_metric = roi_in_metrics(roi=roi, weights=weights)

    #if so, add line to indicate where metric is taken
    if is_metric is not False:
        draw_metric_line(
            roi=roi, 
            weights=weights, 
            ax=ax_lines)
        
        #finish plot
        ax_lines.legend()
    
    ax_lines.set_title(roi)
    ax_lines.set_xlabel('Dose (cGy)')
    ax_lines.set_ylabel('Rel. Volume')   

    #save figure if desired
    if plot_folder is not None: 
        if filter_dict is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/dvh_scatter_{roi}.png', dpi=200)

    elif return_fig:
        return fig


def full_metric_dvh_plot(
    plans: List[TreatmentPlan],
    plot_folder=None
    ) -> None:
    """
    Create figure containing total cost scatter and dvh for each roi
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to plot DVHs for.
    plot_folder (str, optional): Folder to save the plot. If None, the plot is showed but not saved.
    """
    #get example patient
    pat = plans[0].patient
    rois = pat.roi_names
    #for each metric, create separate plot showing scatter of relevant cost on the left and DVHs of corresponding ROI on the right
    for roi in rois:
        fig = dvh_scatter_plot(roi=roi)
        fig.suptitle(self.patient_id)
        plt.savefig(f'{plot_folder}/dvh_{roi}.png', dpi=200)
    

def single_beam_filtered_plot(
    plans: List[TreatmentPlan],
    old_plans: List[TreatmentPlan],
    filter_dict: dict,
    plot_folder:str=None,
    ) -> None:
    """
    Create figure containing total cost scatter and dvh for each roi in a single figure for filtered plans. Apply AFTER filtering.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to plot DVHs for.
    old_plans (List[TreatmentPlan]): Complete list of TreatmentPlan objects if filter was applied.
    filter_dict (dict): Dictionary containing filter information for filtered gaze angles.
    plot_folder (str, optional): Folder to save the plot. If None, the plot is showed but not saved.
    """

    #example patient
    patient = plans[0].patient
    rois = patient.roi_names

    fig = plt.figure(figsize=(14,10), constrained_layout=True)

    ax_scatters = fig.add_subplot(3, 4, 1, projection='polar')
    single_gaze_plot(metric='total_cost', plans=plans, ax=ax_scatters)
    ax_scatters.set_title('Total Cost')
    ax_scatters.legend()

    #plot dvhs for each roi in remaining subplots

    for i, roi in enumerate(rois):
        ax_lines = fig.add_subplot(3, 4, i+2)
        dvh_scatter_plot(
            plans=plans, 
            roi=roi, 
            ax_lines=ax_lines, 
            old_plans=old_plans,
            filter_dict=filter_dict
            )

    fig.suptitle(patient.patient_id)

    #save figure if desired
    if plot_folder is not None: 
        if filtered_gaze_angle_keys is not None:
            name = filter_dict['name']
            plot_folder = f'{plot_folder}/filtered/{name}/'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plt.savefig(f'{plot_folder}/dvhs.png', dpi=200)


def compare_dvhs(self, ray_path, plot_folder):
    """
    Compares selfcalculated 
    """
    ray_dvhs = get_ray_dvh(path=ray_path)
    plt.figure()
    for i, roi_name in enumerate(self.roi_names):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        dose_ray = np.array(ray_dvhs[roi_name]['dose'])*100
        vol_ray = ray_dvhs[roi_name]['volume']

        my_dose, my_vol = self.gaze_angle_dvhs['(0, 0)'].roi_dvhs[roi_name].get_dvh()

        plt.plot(dose_ray/100, vol_ray, linestyle='--', color=color)
        plt.plot(my_dose, my_vol, color=color)
    
    plt.plot([], [], 'k--', label='RayStation DVH')
    plt.plot([], [], 'k-', label='Calculated DVH')
    plt.title('DVH Comparison')
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(f'{plot_folder}/{self.patient_id}.png', dpi=200)


import numpy as np
import h5py
import pandas as pd
import ast
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from patient_functions.helpers import cumulative_dvh, print_progress_bar

class Metric:
    def __init__(self, metric_str): #D20_Macula
        self.name = metric_str
        what, self.roi = metric_str.split('_')  #split into D20 and Macula
        self.metric_type = what[0]  #D
        self.value = int(what[1:])  #20

class Weight:
    def __init__(self, metric: Metric, value: float):
        self.metric = metric
        self.value = value
    
    def __str__(self):
        return f'{self.metric.name}: {self.weight}'

class Weights:
    def __init__(self, weights_dict):
        self.weights = [Weight(Metric(metric_str), weight) for metric_str, weight in weights_dict.items()]
    
    def __str__(self):
        return ', '.join([f'{weight.metric.name}: {weight.value}' for weight in self.weights])
    
    def __iter__(self):
        return iter(self.weights)
    
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, index):
        return self.weights[index]
    
    def weight_for_roi(self, roi, default=1):
        for weight in self.weights:
            if weight.metric.roi == roi:
                return weight.value
        return default

class Filter:
    def __init__(self, metric_str, max_value): #{'D90_Macula': 40}
        """
        Defines a filter where the specified metric must be less than max_value.
        metric_str: str, e.g. 'D90_Macula'
        max_value: float, e.g. 40
        """
        self.metric = Metric(metric_str)
        self.max_value = max_value
    
    def __str__(self):
        return f'{self.metric.name} < {self.max_value}'

class Filters:
    def __init__(self, filter_dict):
        """
        Defines multiple filters.
        filter_dict: dict, e.g. {'D90_Macula': 40, 'D20_Retina': 15}
        """
        self.filters = [Filter(metric_str, max_value) for metric_str, max_value in filter_dict.items()]
    
    def __str__(self):
        return ', '.join([f'{filter.metric.name} < {filter.max_value}' for filter in self.filters])
    
    def __iter__(self):
        return iter(self.filters)
    
    def __len__(self):
        return len(self.filters)
    
    def __getitem__(self, index):
        return self.filters[index]
    
class TreatmentPlan:
    def __init__(
        self,
        patient,
        angle_key,
        dose,
        angle_key_2=None,
        beam_weight=None,
    ):  
        self.name = angle_key if angle_key_2 is None else f"{np.round(beam_weight, 2)}*{angle_key} + {np.round(1-beam_weight, 2)}*{angle_key_2}"
        self.patient = patient
        self.angle_key = angle_key
        if angle_key_2 is not None or beam_weight is not None:

            if angle_key_2 is not None and beam_weight is not None:
                self.beam_weight = beam_weight
                self.angle_key_2 = angle_key_2
            else:
                raise ValueError(f"Either both angle_key_2 ({angle_key_2}) and beam_weight ({beam_weight}) have to be specified or none.")

        self.roi_names = patient.roi_names
        self.dvhs = {}
        for roi_name in self.roi_names:
            self.dvhs[roi_name] = DVH(patient, roi_name, dose)
    

    def __str__(self):
        return f'Treatment Plan for Patient {self.patient.patient_id}, Angle Key: {self.angle_key}, Second Angle Key: {self.angle_key_2}, Weight: {self.beam_weight}'
    
    def calculate_cost(self):
        metric_term = self.calculate_metric_term()
        volume_term = self.calculate_volume_term()
        self.cost = metric_term + volume_term
        return self.cost

    def calculate_volume_term(self):
        volume_term = 0
        for roi_name in self.patient.roi_names:
            w = self.patient.weights.weight_for_roi(roi_name)
            x = self.dvhs[roi_name].get_dvh_auc()/100
            volume_term += w * x
        return volume_term
    
    def calculate_metric_term(self, output_contributions=False):
        contributions = {}
        metric_term = 0

        #iterate over all weights
        for weight in self.patient.weights:
            
            #if metric is D_v
            if weight.metric.metric_type == 'D':
                metric_value = self.dvhs[weight.metric.roi].get_dose_at_volume(volume=weight.value)
            
            #if metric is V_d
            else: 
                metric_value = self.dvhs[weight.metric.roi].get_volume_at_dose(dose=weight.value)
            
            #calculate contribution
            contribution = weight.value * metric_value
            contributions[weight.metric.name] = contribution
            metric_term += contribution

        #return contributions if requested
        if output_contributions:
            return metric_term, contributions

        #otherwise return just metric term
        return metric_term
    
    def calculate_contributions(self):
        _, contributions = self.calculate_metric_term(output_contributions=True)
        contributions['Volume Term'] = self.calculate_volume_term()
        return contributions
    
    def plot_dvhs(self, ax):
        for roi in self.roi_names:
            self.dvhs[roi].plot(ax, plot_args={'label': roi})
        ax.legend()
        ax.grid()

class DVH:
    def __init__(
        self,
        patient,
        roi_name,
        dose
        ):
        self.patient_id = patient.patient_id
        self.roi_name = roi_name
        self.dose, self.volume = cumulative_dvh(
            dose=dose[patient.roi_masks[self.roi_name]], 
            frac=patient.roi_relative_values[self.roi_name], 
            voxel_vol = patient.voxel_vol, 
            bins=patient.num_dvh_bins)


    def __str__(self):
        return f'DVH for Patient {self.patient_id}, ROI: {self.roi_name}'

    def plot(self, ax, plot_args={}):
        ax.plot(self.dose, self.volume, **plot_args)
    
    def get_dvh(self):
        return self.dose, self.volume
    
    def get_volume_at_dose(self, dose):
        """
        Return Vx: the volume (%) receiving at least dose.
        dvh_dose, dvh_volume can be ascending or descending; handles both.
        """
        # Ensure numpy arrays
        dvh_dose = np.asarray(self.dose)
        dvh_volume = np.asarray(self.volume)
        
        # Make sure dose is ascending for interpolation
        if dvh_dose[0] > dvh_dose[-1]:
            dvh_dose = dvh_dose[::-1]
            dvh_volume = dvh_volume[::-1]
        
        # Interpolate volume at dose x
        return np.interp(dose, dvh_dose, dvh_volume)

    def get_dose_at_volume(self, volume):
        """
        Return Dx: the dose corresponding to volume x.
        - x can be scalar or array (volume units).
        - If clip=True (default), x outside the range of `volume` is clipped to min/max.
        If clip=False, np.interp will extrapolate using end values (which is usually undesirable).
        Assumes `volume` is cumulative DVH (monotonic, typically decreasing with dose).
        """
        dvh_dose = np.asarray(self.dose)
        dvh_volume = np.asarray(self.volume)

        # np.interp requires the xp (here: volume) to be ascending.
        # If volume is descending, reverse both arrays to make volume ascending.
        if dvh_volume[0] > dvh_volume[-1]:
            dvh_dose = dvh_dose[::-1]
            dvh_volume = dvh_volume[::-1]
        volume = np.asarray(volume)
        volume = np.clip(volume, dvh_volume.min(), dvh_volume.max())
        # interpolate dose as a function of volume (xp=volume, fp=dose)
        return np.interp(volume, dvh_volume, dvh_dose)
    
    def get_dvh_auc(self):
        return np.trapz(y=self.volume, x=self.dose)
    
    def get_metric_value(self, metric: Metric):
        if self.roi_name != metric.roi_name:
            raise ValueError(f"Metric ROI '{metric.roi_name}' does not match DVH ROI '{self.roi_name}'")

        if metric.metric_type == 'D':
            return self.get_dose_at_volume(metric.value)
        else: return self.get_volume_at_dose(metric.value)

class Patient:
    def __init__(
        self, 
        patient_id, 
        h5_file_path, 
        num_dvh_bins=200,
        weights={'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1},
        ):
        print(f'Initializing Patient {patient_id}...\n', end='\r')
        self.patient_id = patient_id
        self.h5_file_path = h5_file_path   
        self.weights = Weights(weights)
        self.num_dvh_bins = num_dvh_bins

        #extract gaze angles, metrics, costs, and dvh data from h5py file
        with h5py.File(self.h5_file_path, "r") as h5_file:
            f_keys = h5_file.keys()
            self.gaze_angle_keys = [key for key in f_keys if '(' in key]
            self.gaze_angles = np.array([ast.literal_eval(item) for item in self.gaze_angle_keys])
            self.roi_names = h5_file.attrs['roi_names']
            self.roi_names = [roi for roi in self.roi_names if roi.lower() != 'tumor']
            self.roi_masks = {roi_name: h5_file[f'{roi_name}_mask'][:] for roi_name in self.roi_names}
            self.roi_relative_values = {roi_name: h5_file[f'{roi_name}_relative_volumes'][:] for roi_name in self.roi_names}
            self.voxel_vol = h5_file.attrs['voxel_volume']

            self.gaze_angle_dvhs = {}
            for gaze_angle_key in self.gaze_angle_keys:
                dose = h5_file[gaze_angle_key][:]
                self.gaze_angle_dvhs[gaze_angle_key] = TreatmentPlan(
                                                            patient=self,
                                                            angle_key=gaze_angle_key,
                                                            dose=dose
                                                            )

        #extract polar and azimuthal angles and theta for plotting
        self.polar = self.gaze_angles[:,0]
        self.azimuthal = self.gaze_angles[:,1]
        self.theta = np.deg2rad(self.azimuthal) #used for polar plots
    
    def __str__(self):
        return f'Patient {self.patient_id} with {len(self.gaze_angle_keys)} gaze angles and ROIs: {", ".join(self.roi_names)}'

def find_best_beam_weight(patient, gaze_angle_key_1, gaze_angle_key_2, full_output=False, n_steps=10):
    ws = np.linspace(0, 1, n_steps)
    costs = []
    plans = []
    with h5py.File(patient.h5_file_path, "r") as h5_file:
        dose_1 = h5_file[gaze_angle_key_1][:]
        dose_2 = h5_file[gaze_angle_key_2][:]
    for w in ws:
        combined_dose = w*dose_1 + (1-w)*dose_2
        
        g = TreatmentPlan(
            patient=patient,
            angle_key=gaze_angle_key_1,
            angle_key_2=gaze_angle_key_2,
            dose=combined_dose,
            beam_weight=w
        )
        cost = g.calculate_cost()
        plans.append(g)
        costs.append(cost)
        del g

    costs = np.asarray(costs)
    opt_idx = np.argmin(costs)
    opt_cost = costs[opt_idx]
    opt_w = ws[opt_idx]
    opt_dvh = plans[opt_idx]
    if full_output: return ws, costs, plans
    else: 
        return opt_w, opt_cost, opt_dvh

def plot_weight_search(patient, gaze_angle_key_1, gaze_angle_key_2, n_steps=10):
    n_plots = len(patient.roi_names)
    ws, costs, plans = find_best_beam_weight(
        patient=patient,
        gaze_angle_key_1=gaze_angle_key_1,
        gaze_angle_key_2=gaze_angle_key_2,
        full_output=True, 
        n_steps=n_steps
        )
    
    opt_idx = np.argmin(np.asarray(costs))
    opt_cost = costs[opt_idx]
    opt_w = ws[opt_idx]
    
    norm = plt.Normalize(min(costs), max(costs))
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(4, 3, figsize=(10,10), constrained_layout=True)
    plt.suptitle(f'{gaze_angle_key_1} + {gaze_angle_key_2} ({patient.patient_id})')
    ax = axes.flatten()[0]
    ax.scatter(ws, costs, c=costs, cmap=cmap, norm=norm, s=80)
    ax.scatter(opt_w, opt_cost, color='red', s=100)
    ax.set_title(f'Weights vs. Costs')
    ax.set_xlabel("Weight")
    ax.set_ylabel("Cost")

    plan_1 = plans[-1]
    plan_2 = plans[0]

    for roi_name, ax in zip(patient.roi_names, axes.flatten()[1:]):
        for w, cost, plan in zip(ws, costs, plans):
            color = 'red' if cost == opt_cost else cmap(norm(cost)) 
            z = 10 if cost==opt_cost else 1
            alpha = 1 if cost==opt_cost else 0.5
            plan.dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': z, 'alpha': alpha})

            ax.set_title(roi_name)
            ax.set_xlabel("Dose (Gy)")
            ax.set_ylabel("Volume (%)")
        
        plan_1.dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{gaze_angle_key_1}', 'zorder': 2, 'lw': 1})
        plan_2.dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{gaze_angle_key_2}', 'zorder': 2, 'lw': 1})
    plt.legend()
    plt.show()
    return plans

def plot_gaze_combo_heatmap(costs, beam_weights, gaze_angle_keys, ax):

    mask = np.triu(np.ones_like(costs, dtype=bool), k=1).T
    finite_vals = costs[np.isfinite(costs)]
    vmin = finite_vals.min()
    vmax = finite_vals.max()
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mask_diag = np.eye(beam_weights.shape[0], dtype=bool)
    beam_weights = np.round(beam_weights, 2).astype(str)
    beam_weights[mask_diag] = ""  # empty string to skip annotation

    min_pos = np.unravel_index(np.argmin(costs), costs.shape)
    row, col = min_pos

    # Draw red rectangle around the cell
    rect = mpl.patches.Rectangle((col, row), 1, 1, fill=False, edgecolor='red', linewidth=3)
    
    sns.heatmap(data=costs, annot=beam_weights, mask=mask, xticklabels=gaze_angle_keys, yticklabels=gaze_angle_keys, ax=ax, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax, cbar_kws={'label': 'Cost'}, fmt="")
    ax.add_patch(rect)
    return cmap, norm

def calculate_gaze_combos(patient, n_steps=10):
    gaze_angle_keys = patient.gaze_angle_keys
    print("Calculating all gaze angle combinations...")
    n = len(gaze_angle_keys)
    all_costs= []
    all_beam_weights = []
    all_plans = []

    for i, angle_1 in enumerate(gaze_angle_keys):
        cost_row = [np.inf]*i
        weight_row = [np.inf]*i
        plan_row = [False]*i
        for j, angle_2 in enumerate(gaze_angle_keys[i:]):
            opt_w, opt_cost, opt_plan = find_best_beam_weight(
                patient, 
                gaze_angle_key_1=angle_1, 
                gaze_angle_key_2=angle_2)

            cost_row.append(opt_cost)
            weight_row.append(opt_w)
            plan_row.append(opt_plan)

        all_costs.append(cost_row)
        all_beam_weights.append(weight_row)
        all_plans.append(plan_row)
        print_progress_bar(i+1, n)
    return np.asarray(all_beam_weights), np.asarray(all_costs), np.asarray(all_plans)

def plot_all_gaze_combos(patient, n_steps=10):
    gaze_angle_keys = patient.gaze_angle_keys
    beam_weights, costs, plans = calculate_gaze_combos(patient, n_steps=n_steps)
    n_angles = len(gaze_angle_keys)

    fig, axes = plt.subplots(4, 3, figsize=(12,12), constrained_layout=True)
    axes = axes.flatten()
    cmap, norm = plot_gaze_combo_heatmap(
        costs=costs, 
        beam_weights=beam_weights, 
        gaze_angle_keys=gaze_angle_keys, 
        ax=axes[0])


    opt_idx = np.unravel_index(costs.argmin(), costs.shape)
    opt_cost = costs[opt_idx]
    opt_w = beam_weights[opt_idx]
    opt_plan = plans[opt_idx[0]][opt_idx[1]]

    angle_1 = opt_plan.angle_key
    angle_2 = opt_plan.angle_key_2
    
    with h5py.File(patient.h5_file_path, "r") as h5_file:
            dose_1 = h5_file[angle_1][:]
            dose_2 = h5_file[angle_2][:]
        
    plan_1 = TreatmentPlan(patient=patient, angle_key=angle_1, dose=dose_1)
    plan_2 = TreatmentPlan(patient=patient, angle_key=angle_2, dose=dose_2)
    print(plans)
    for roi_name, ax in zip(plan_1.roi_names, axes[1:]):
        for w, cost, plan in zip(beam_weights.flatten(), costs.flatten(), plans.flatten()):
            if plan is not False:
                if cost == opt_cost:
                    plan.dvhs[roi_name].plot(ax=ax, plot_args={'color': 'red', 'zorder': 3, 'alpha': 1, 'label': f'{np.round(opt_w,2)}*{angle_1} + {np.round(1-opt_w, 2)}*{angle_2}'})
                else:
                    color=cmap(norm(cost))
                    plan.dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': 1, 'alpha': 0.01})

        plan_1.dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{angle_1}', 'zorder': 4, 'lw': 1})
        plan_2.dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{angle_2}', 'zorder': 4, 'lw': 1})
        ax.set_title(roi_name)
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.grid()
    ax.legend()
    #plt.savefig(f'plots/{self.patient.patient_id}/two_beams/find_optimal_combination_{n_angles}_angles.png', dpi=300)
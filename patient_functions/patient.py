import numpy as np
import h5py
import pandas as pd
from scipy.optimize import minimize_scalar
import ast
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Rectangle


from patient_functions.plotting import *
from patient_functions.filter_dvh import *
from patient_functions.dvh import *
from patient_functions.cost import *
from patient_functions.helpers import *
from patient_functions.gaze_combo import *


import time

class TwoBeam:
    def __init__(
        self,
        patient,
        gaze_angle_key_1,
        gaze_angle_key_2,
        weight=0.5,
    ):  

        self.patient = patient.h5py_file_path
        self.gaze_angle_key_1 = gaze_angle_key_1
        self.gaze_angle_key_2 = gaze_angle_key_2
        self.weight = weight
        with h5py.File(self.h5py_file_path, "r") as h5_file:
            self.dose_1 = h5_file[self.gaze_angle_key_1][:]
            self.dose_2 = h5_file[self.gaze_angle_key_2][:]
        self.combined_dose = self.weight*self.dose_1 + (1-self.weight)*self.dose_2
        self.dvhs = GazeAngleDVHs(
            patient=self.patient, 
            angle_key=self.gaze_angle_key_1,
            angle_key_2=self.gaze_angle_key_2,
            dose=self.combined_dose)

    def update_doses(self):
        with h5py.File(self.h5py_file_path, "r") as h5_file:
            self.dose_1 = h5_file[self.gaze_angle_key_1][:]
            self.dose_2 = h5_file[self.gaze_angle_key_2][:]
        self.combined_dose = self.weight*self.dose_1 + (1-self.weight)*self.dose_2

    def set_new_gaze_angles(self, gaze_angle_keys):
        self.gaze_angle_key_1 = gaze_angle_keys[0]
        self.gaze_angle_key_2 = gaze_angle_keys[1]
        self.update_doses()
        self.update_dvhs()
        

    def calculate_dose(self):
        return self.weight*self.dose_1 + (1-self.weight)*self.dose_2
    
    def set_new_weight(self, new_weight):
        self.weight = new_weight
        self.combined_dose = self.calculate_dose()
    
    def update_dvhs(self):
        self.dvhs = GazeAngleDVHs(
            patient=self.patient, 
            angle_key=self.gaze_angle_key_1,
            angle_key_2=self.gaze_angle_key_2,
            dose=self.combined_dose,
            weight=self.weight)
    
    def optimize_weight(self):
        def cost_wrapper(w):
            self.set_new_weight(w)
            self.update_dvhs()
            cost = self.dvhs.calculate_cost()
            return cost
        
        res = minimize_scalar(
            fun=cost_wrapper,
            bounds=(0, 1),
        )
        return res
    
    def full_weight_search(self, full_output=False, n_steps=10):
        ws = np.linspace(0, 1, n_steps)
        costs = []
        gaze_angle_dvhs = []
        for w in ws:
            self.set_new_weight(w)
            self.update_dvhs()
            cost = self.dvhs.calculate_cost()
            gaze_angle_dvhs.append(self.dvhs)
            costs.append(cost)
        costs = np.asarray(costs)
        opt_idx = np.argmin(costs)
        opt_cost = costs[opt_idx]
        opt_w = ws[opt_idx]
        opt_dvh = gaze_angle_dvhs[opt_idx]
        if full_output: return ws, costs, gaze_angle_dvhs
        else: 
            return opt_w, opt_cost, opt_dvh
    
    def plot_weight_search(self, n_steps=10):
        n_plots = len(self.patient.roi_names)
        ws, costs, gaze_angle_dvhs = self.full_weight_search(full_output=True, n_steps=n_steps)
        opt_idx = np.argmin(np.asarray(costs))
        opt_cost = costs[opt_idx]
        opt_w = ws[opt_idx]
        
        norm = plt.Normalize(min(costs), max(costs))
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(4, 3, figsize=(10,10), constrained_layout=True)
        plt.suptitle(f'{self.gaze_angle_key_1} + {self.gaze_angle_key_2} ({self.patient.patient_id})')
        ax = axes.flatten()[0]
        ax.scatter(ws, costs, c=costs, cmap=cmap, norm=norm, s=80)
        ax.scatter(opt_w, opt_cost, color='red', s=100)
        ax.set_title(f'Weights vs. Costs')
        ax.set_xlabel("Weight")
        ax.set_ylabel("Cost")

        dvhs_1 = self.patient.gaze_angle_dvhs[self.gaze_angle_key_1]
        dvhs_2 = self.patient.gaze_angle_dvhs[self.gaze_angle_key_2]

        for roi_name, ax in zip(self.patient.roi_names, axes.flatten()[1:]):
            for w, cost, gaze_angle_dvh in zip(ws, costs, gaze_angle_dvhs):
                color = 'red' if cost == opt_cost else cmap(norm(cost)) 
                z = 10 if cost==opt_cost else 1
                alpha = 1 if cost==opt_cost else 0.5
                gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': z, 'alpha': alpha})

                ax.set_title(roi_name)
                ax.set_xlabel("Dose (Gy)")
                ax.set_ylabel("Volume (%)")
            
            dvhs_1.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{self.gaze_angle_key_1}', 'zorder': 2, 'lw': 1})
            dvhs_2.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{self.gaze_angle_key_2}', 'zorder': 2, 'lw': 1})
        plt.legend()
        #plt.savefig(f'plots/{self.patient.patient_id}/two_beams/{self.gaze_angle_key_1}_{self.gaze_angle_key_2}_dvhs.png', dpi=200)
    
    def plot_gaze_combo_heatmap(self, costs, weights, ax):

        mask = np.triu(np.ones_like(costs, dtype=bool), k=1).T
        finite_vals = costs[np.isfinite(costs)]
        vmin = finite_vals.min()
        vmax = finite_vals.max()
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mask_diag = np.eye(weights.shape[0], dtype=bool)
        weights = np.round(weights, 2).astype(str)
        weights[mask_diag] = ""  # empty string to skip annotation

        min_pos = np.unravel_index(np.argmin(costs), costs.shape)
        row, col = min_pos

        # Draw red rectangle around the cell
        rect = Rectangle((col, row), 1, 1, fill=False, edgecolor='red', linewidth=3)
        
        sns.heatmap(data=costs, annot=weights, mask=mask, xticklabels=self.patient.gaze_angle_keys, yticklabels=self.patient.gaze_angle_keys, ax=ax, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax, cbar_kws={'label': 'Cost'}, fmt="")
        ax.add_patch(rect)
        return cmap, norm
                
                
    def calculate_gaze_combos(self, n_steps=10):
        print("Calculating all gaze angle combinations...")
        n = len(self.patient.gaze_angle_keys)
        all_costs= []
        all_weights = []
        all_gaze_angle_dvhs = []

        for i, angle_1 in enumerate(self.patient.gaze_angle_keys):
            cost_row = [np.inf]*i
            weight_row = [np.inf]*i
            dvh_row = [False]*i
            for j, angle_2 in enumerate(self.patient.gaze_angle_keys[i:]):
                self.set_new_gaze_angles((angle_1, angle_2))
                w, cost, dvh = self.full_weight_search(n_steps=n_steps)
                cost_row.append(cost)
                weight_row.append(w)
                dvh_row.append(dvh)
            all_costs.append(cost_row)
            all_weights.append(weight_row)
            all_gaze_angle_dvhs.append(dvh_row)
            print_progress_bar(i+1, n)
        return np.asarray(all_weights), np.asarray(all_costs), np.asarray(all_gaze_angle_dvhs)

    def plot_all_gaze_combos(self, n_steps=10):
        weights, costs, dvhs = self.calculate_gaze_combos(n_steps=n_steps)
        n_angles = len(self.patient.gaze_angle_keys)

        fig, axes = plt.subplots(4, 3, figsize=(12,12), constrained_layout=True)
        axes = axes.flatten()
        cmap, norm = self.plot_gaze_combo_heatmap(costs, weights, axes[0])

        opt_idx = np.unravel_index(costs.argmin(), costs.shape)
        opt_cost = costs[opt_idx]
        opt_w = weights[opt_idx]
        opt_dvh = dvhs[opt_idx[0]][opt_idx[1]]

        angle_1 = opt_dvh.angle_key
        angle_2 = opt_dvh.angle_key_2
        dvhs_1 = self.patient.gaze_angle_dvhs[angle_1]
        dvhs_2 = self.patient.gaze_angle_dvhs[angle_2]

        for roi_name, ax in zip(self.patient.roi_names, axes[1:]):
            for w, cost, gaze_angle_dvh in zip(weights.flatten(), costs.flatten(), dvhs.flatten()):
                if gaze_angle_dvh is not False:
                    if cost == opt_cost:
                        gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'red', 'zorder': 3, 'alpha': 1, 'label': f'{np.round(opt_w,2)}*{angle_1} + {np.round(1-opt_w, 2)}*{angle_2}'})
                    else:
                        color=cmap(norm(cost))
                        #gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': 1, 'alpha': 0.01})

            dvhs_1.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{angle_1}', 'zorder': 4, 'lw': 1})
            dvhs_2.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{angle_2}', 'zorder': 4, 'lw': 1})
            ax.set_title(roi_name)
            ax.set_xlabel("Dose (Gy)")
            ax.set_ylabel("Volume (%)")
            ax.grid()
        ax.legend()
        plt.savefig(f'plots/{self.patient.patient_id}/two_beams/find_optimal_combination_{n_angles}_angles.png', dpi=300)
        

class RoiDVH:
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

    def plot(self, ax, plot_args={}):
        ax.plot(self.dose, self.volume, **plot_args)
    
    def get_dvh(self):
        return self.dose, self.volume
    
    def get_volume_at_dose(self, dose):
        """
        #Return Vx: the volume (%) receiving at least dose.
        #dvh_dose, dvh_volume can be ascending or descending; handles both.
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
    
    def get_metric_value(self, metric):
        if metric['metric_type'] == 'D':
            return self.get_dose_at_volume(metric['value'])
        else: return self.get_volume_at_dose(metric['value'])


class GazeAngleDVHs:
    def __init__(
        self,
        patient,
        angle_key,
        dose,
        angle_key_2=None,
        weight=None,
    ):  
        self.patient = patient
        self.angle_key = angle_key
        self.is_two_beam = True if self.angle_key_2 is not None else False
        self.weight = weight
        if angle_key_2 is not None and weight is not None:
            self.weight = weight
            self.angle_key_2 = angle_key_2
        
        self.roi_dvhs = {}
        for roi_name in patient.roi_names:
            self.roi_dvhs[roi_name] = RoiDVH(patient, roi_name, dose)
    
    def calculate_cost(self):
        metric_term = self.calculate_metric_term()
        volume_term = self.calculate_volume_term()
        self.cost = metric_term + volume_term
        return self.cost

    def calculate_volume_term(self):
        volume_term = 0
        for roi_name in self.patient.roi_names:
            w = next((item["weight"] for item in self.patient.weights if item["roi_name"] == roi_name), 1)
            x = self.roi_dvhs[roi_name].get_dvh_auc()/100
            volume_term += w * x
        return volume_term
    
    def calculate_metric_term(self):
        metric_term = 0
        for i, weight in enumerate(self.patient.weights):
            #print(weight)
            if weight['metric_type'] == 'D':
                metric_value = self.roi_dvhs[weight['roi_name']].get_dose_at_volume(volume=weight['value'])
            
            else: 
                metric_value = self.roi_dvhs[weight['roi_name']].get_volume_at_dose(dose=weight['value'])

            metric_term += weight['weight'] * metric_value
        return metric_term

class Patient:
    def __init__(
        self, 
        patient_id, 
        h5py_file_path, 
        num_dvh_bins=1000,
        weights={'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1},
        ):
        print(f'Initializing Patient {patient_id}...\n', end='\r')
        self.patient_id = patient_id
        self.h5py_file_path = h5py_file_path   
        self.weights = convert_weights(weights)
        self.num_dvh_bins = num_dvh_bins

        #extract gaze angles, metrics, costs, and dvh data from h5py file
        with h5py.File(self.h5py_file_path, "r") as h5_file:
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
                self.gaze_angle_dvhs[gaze_angle_key] = GazeAngleDVHs(
                                                            patient=self,
                                                            angle_key=gaze_angle_key,
                                                            dose=dose
                                                            )
                                                    
        #extract polar and azimuthal angles and theta for plotting
        self.polar = self.gaze_angles[:,0]
        self.azimuthal = self.gaze_angles[:,1]
        self.theta = np.deg2rad(self.azimuthal) #used for polar plots


    
    def set_weights(self, weights):
        self.weights = convert_weights(weights)
    

    def save_all_gaze_combos(self, n_steps=10):
        gaze_angle_dvhs = []
        for i, angle_1 in enumerate(self.patient.gaze_angle_keys):
            for j, angle_2 in enumerate(self.patient.gaze_angle_keys[i:]):
                self.set_new_gaze_angles((angle_1, angle_2))
                for w in np.linspace(0, 1, n_steps):
                    self.set_new_weight(w)
                    self.update_dvhs()
                    gaze_angle_dvhs.append(self.dvhs)

    def find_gaze_angle_smaller_vdose(self, dose_distribution, roi, dose, max_vol):
        raise NotImplementedError

    def find_gaze_angle_smaller_dvol(self, roi, vol, max_dose):
        raise NotImplementedError
    
    def find_new_optimal_gaze_angles(self, filtered_gaze_angle_keys):
        raise NotImplementedError
        
    def single_gaze_plot(self, ax, metric, filtered_gaze_angle_keys):
        raise NotImplementedError
    
    def full_scatter_plot(self, plot_folder, filtered_gaze_angle_keys, filter_dict):
        raise NotImplementedError
    
    def plot_gaze_angle_dvhs(self, ax, roi, colors, gaze_angle_keys):
        raise NotImplementedError
    
    def roi_in_metrics(self, roi):
        raise NotImplementedError
    
    def find_metric_for_roi(self, roi):
        raise NotImplementedError
    
    def dvh_scatter_plot(self, roi, gaze_angle_keys):
        raise NotImplementedError
    
    def full_metric_dvh_plot(self, plot_folder):
        raise NotImplementedError
    
    def find_contributions(self, gaze_angle_key):
        raise NotImplementedError
    
    def plot_filtered_dvh(self, roi, gaze_angle_keys):
        raise NotImplementedError
    
    def get_gaze_angles_and_costs_from_keys(self, gaze_angle_keys):
        raise NotImplementedError

    def full_filtered_metric_dvh_plot(self, filter_dict, plot_folder):
        raise NotImplementedError
    
    def compare_contributions_bar(self, gaze_angle_key1, gaze_angle_key2, plot_folder):
        raise NotImplementedError
    
    def test_gaze_combination(self, gaze_angle_keys, gaze_angle_weights, metric):
        raise NotImplementedError

    def get_dose_at_volume(self, dvh, volume):
        raise NotImplementedError
    
    def get_volume_at_dose(self, dvh, dose):
        raise NotImplementedError
    
    def get_metric(self, metric, roi, total_dose):
        raise NotImplementedError

    def cost_1_beam(self, weights, dose):
        raise NotImplementedError
    
    def minimizer(self, dose_1, dose_2):
        raise NotImplementedError
    
    def compare_dvhs(self, ray_path):
        raise NotImplementedError
    
    def apply_dvh_filters(self, dvh_filters):
        raise NotImplementedError
    

    

Patient.find_gaze_angle_smaller_vdose = find_gaze_angle_smaller_vdose
Patient.find_gaze_angle_smaller_dvol = find_gaze_angle_smaller_dvol
Patient.find_new_optimal_gaze_angles = find_new_optimal_gaze_angles
Patient.single_gaze_plot = single_gaze_plot
Patient.full_scatter_plot = full_scatter_plot
Patient.plot_gaze_angle_dvhs = plot_gaze_angle_dvhs
Patient.roi_in_metrics = roi_in_metrics
Patient.find_metric_for_roi = find_metric_for_roi
Patient.dvh_scatter_plot = dvh_scatter_plot
Patient.full_metric_dvh_plot = full_metric_dvh_plot
Patient.find_contributions = find_contributions
Patient.plot_filtered_dvh = plot_filtered_dvh
Patient.get_gaze_angles_and_costs_from_keys = get_gaze_angles_and_costs_from_keys
Patient.full_filtered_metric_dvh_plot = full_filtered_metric_dvh_plot
Patient.compare_contributions_bar = compare_contributions_bar
Patient.test_gaze_combination = test_gaze_combination
Patient.get_dose_at_volume = get_dose_at_volume
Patient.get_volume_at_dose = get_volume_at_dose
Patient.get_metric = get_metric
Patient.cost_1_beam = cost_1_beam
Patient.minimizer = minimizer
Patient.compare_dvhs = compare_dvhs
Patient.apply_dvh_filters = apply_dvh_filters

def main():
    
    patient_id = 'P23336'
    plot_folder = f'plots/{patient_id}'
    pat = Patient(patient_id=patient_id, h5py_file_path=f'results/{patient_id}_5.h5')

    
    start_time = time.time()
    filter_dict = {'name': 'D90_Macula', 'roi': 'Macula', 'filter_type': 'D', 'value': 90, 'max': 40}  # Example filter criteria
    #filtered_keys = pat.full_filtered_metric_dvh_plot(filter_dict=filter_dict, plot_folder=None, save_fig=False)
    #new_opt = pat.find_new_optimal_gaze_angles(filtered_gaze_angle_keys=filtered_keys)['New Optimum']

    metric_weights = [
        {'roi': 'Macula',       'metric': 'D2',     'weight': 3},
        {'roi': 'OpticalDisc',  'metric': 'D20',    'weight': 3},
        {'roi': 'Cornea',       'metric': 'D20',    'weight': 1},
        {'roi': 'Retina',       'metric': 'V55',    'weight': 1},
        {'roi': 'CiliaryBody',  'metric': 'V27',    'weight': 1},
        {'roi': 'Lens',         'metric': 'D5',     'weight': 1}]
    
    with h5py.File(pat.h5py_file_path, "r") as h5_file:
        total_dose = h5_file['gaze_angles'][pat.optimal_angle_key]['total_dose'][:]

    pat.compare_dvhs(ray_path='results/P23336_ray_dvhs.txt', total_dose=total_dose)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')
if __name__ == "__main__":
    main()
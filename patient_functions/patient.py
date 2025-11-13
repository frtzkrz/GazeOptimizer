import numpy as np
import h5py
import pandas as pd
import ast
from patient_functions.plotting import *
from patient_functions.filter_dvh import *
from patient_functions.dvh import *
from patient_functions.cost import *
from patient_functions.helpers import *

import time

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
    ):  
        self.patient = patient
        self.angle_key = angle_key
        self.roi_dvhs = {}
        for roi_name in patient.roi_names:
            self.roi_dvhs[roi_name] = RoiDVH(patient, roi_name, dose)
            
    def calculate_cost(self):
        return self.patient.cost_1_beam(self.angle_key, full_results=False)
    
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
        print(f'Initializing Patient {patient_id}...', end='\r')
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
        self.num_dvh_bins = num_dvh_bins


    
    def set_weights(self, weights):
        self.weights = convert_weights(weights)

        





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
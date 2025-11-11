import numpy as np
import h5py
import pandas as pd
from patient_functions.plotting import *
from patient_functions.filter_dvh import *
from patient_functions.dvh import *
from patient_functions.cost import *

import time

class Patient:
    def __init__(
            self, 
            patient_id, 
            h5py_file_path, 
            num_dvh_bins = 100, 
            weights={'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1},
            ):
        print(f'Initializing Patient {patient_id}...', end='\r')
        self.patient_id = patient_id
        self.h5py_file_path = h5py_file_path   
        self.weights = weights

        #extract gaze angles, metrics, costs, and dvh data from h5py file
        with h5py.File(self.h5py_file_path, "r") as h5_file:
            self.gaze_angle_keys = list(h5_file['gaze_angles'].keys())
            self.gaze_angle_dict = {gaze_angle_key: h5_file['gaze_angles'][gaze_angle_key].attrs['gaze_angle'] for gaze_angle_key in self.gaze_angle_keys}     
            self.angles=np.array([h5_file['gaze_angles'][gaze_angle_key].attrs['gaze_angle'] for gaze_angle_key in self.gaze_angle_keys])
            self.costs = np.array([h5_file['gaze_angles'][gaze_angle_key].attrs['total_cost'] for gaze_angle_key in self.gaze_angle_keys])
            self.costs_dict = {gaze_angle_key: h5_file['gaze_angles'][gaze_angle_key].attrs['total_cost'] for gaze_angle_key in self.gaze_angle_keys}
            self.idx_min = np.argmin(self.costs)
            self.rois = [k for k in list(h5_file['gaze_angles'][self.gaze_angle_keys[0]].keys()) if 'clip' not in k.lower() and 'EP' not in k and k not in ['total_dose', 'Tumor']]
            self.optimal_angle_key = self.gaze_angle_keys[self.idx_min]
            self.optimal_metrics = {weight: h5_file['gaze_angles'][self.optimal_angle_key].attrs[weight] for weight in self.weights}
            self.metrics = list(h5_file['gaze_angles'][self.gaze_angle_keys[0]].attrs.keys())
            self.metrics_dict = {gaze_angle_key: {metric: h5_file['gaze_angles'][gaze_angle_key].attrs[metric] for metric in self.metrics} for gaze_angle_key in self.gaze_angle_keys}
            self.roi_mask_dict = {roi: h5_file['roi_masks'][roi][:] for roi in self.rois}

            self.dvh_dict = {}
            for roi in self.rois:
                self.dvh_dict[roi] = {}
                for gaze_angle_key in self.gaze_angle_keys:
                    dvh = np.array(h5_file['gaze_angles'][gaze_angle_key][roi][:])
                    self.dvh_dict[roi][gaze_angle_key] = dvh

        #extract polar and azimuthal angles and theta for plotting
        self.polar=self.angles[:,0]
        self.azimuthal=self.angles[:,1]
        self.theta=np.deg2rad(self.azimuthal) #used for polar plots

        #find optimal gaze angles based on total cost
        
        
        self.optimal_polar=self.polar[self.idx_min]
        self.optimal_azimuthal=self.azimuthal[self.idx_min]
        self.optimal_theta=self.theta[self.idx_min]
        self.optimal_cost=self.costs[self.idx_min]
    
        self.num_dvh_bins = num_dvh_bins



    def find_gaze_angle_smaller_vdose(self, roi, dose, max_vol):
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
    
    def cumulative_dvh(self, dose_array, num_points=100):
        raise NotImplementedError

    def get_dose_at_volume(self, dvh, volume):
        raise NotImplementedError
    
    def get_volume_at_dose(self, dvh, dose):
        raise NotImplementedError
    
    def get_metric(self, metric, roi, total_dose):
        raise NotImplementedError

    def cost(self, weights, dose):
        raise NotImplementedError
    
    def minimizer(self, dose_1, dose_2):
        raise NotImplementedError
    
    def compare_dvhs(self, ray_path):
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
Patient.cumulative_dvh = cumulative_dvh
Patient.get_dose_at_volume = get_dose_at_volume
Patient.get_volume_at_dose = get_volume_at_dose
Patient.get_metric = get_metric
Patient.cost = cost
Patient.minimizer = minimizer
Patient.compare_dvhs = compare_dvhs

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
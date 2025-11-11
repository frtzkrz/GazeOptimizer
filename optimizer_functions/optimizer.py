from functions.RSAdapt.setDefaultPaths import setDefaultPaths
setDefaultPaths()
import sys
sys.path.append(r"C:\Program Files\RaySearch Laboratories\RayStation 2023B-R\ScriptClient")

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from optimizer_functions.ray_wrappers import *
from optimizer_functions.cost_functions import *
from optimizer_functions.helpers import *
from optimizer_functions.grid_search import *
from optimizer_functions.data_storage import *
from optimizer_functions.plotting import *
from optimizer_functions.save_dose_distributions import *
from optimizer_functions.full_search import *

class GazeOptimizer:
    def __init__(
        self,
        patient_id, 
        weights={'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1},
        maxfev=30,
        delta_polar=5,
        max_polar_deg=25,
        save_path=None,
        prescribed_dose=60,
        clinical_path=None,
        use_precalculated=False,
        h5py_file_path = None,
        num_dvh_bins = 100,
        plot_path='../plots/'):
        
        self.columns = [('gaze_angles', 'polar'), ('gaze_angles', 'azimuthal')] + [('dvh_points', k) for k in weights.keys()] + [('dvh_points', 'dose_volume_penalty'), ('total_cost', 'cost')]
        self.patient_id = patient_id
        self.maxfev = maxfev
        self.delta_polar = delta_polar
        self.clinical_path = clinical_path
        self.prescribed_dose = prescribed_dose
        self.weights = weights
        self.max_polar_deg = max_polar_deg
        self.bounds = [(0,max_polar_deg), (0, 360)]
        self.use_precalculated = use_precalculated
        self.num_dvh_bins = num_dvh_bins
        self.cache={}
        self.plot_folder=f'plots/{self.patient_id}'

        if save_path: self.save_path = save_path
        else: self.save_path = f"results/{self.patient_id}_eval_history.csv"
        
        if h5py_file_path: self.h5py_file_path = h5py_file_path 
        else: self.h5py_file_path = f'results/{self.patient_id}.h5'

        if self.patient_id != self.get_patient_id():
            raise Exception(f'Check Patient ID: Raysearch: {self.get_patient_id()}, while here {self.patient_id}')

    

    def minimizer(self, initial_guess):
        res = minimize(fun=self.cost, bounds=self.bounds, method='Nelder-Mead', x0=initial_guess, options={'maxfev': self.maxfev, 'xatol': 0.5})
        print(f"Success: {res.success} \nafter {res.nfev} evaluations")
        return res

    def optimize(self):
        """
        Optimizes gaze angle:
            First performs a grid search with parameters given when creating instance of GazeOptimizer
            The gaze angle with minimum cost from grid search is used as initial guess for scipy.optimize.minimize
        
        returns: (polar_optimal, azimuthal_optimal)
        """

        gaze_angles = self.define_gaze_angle_grid()
        roi_names = self.get_roi_names()

        with h5py.File(self.h5py_file_path, 'a') as f:

            #Store all roi masks for total dose if not yet saved
            if 'roi_masks' not in f.keys():
                mask_group = f.create_group('roi_masks')
                for roi in roi_names:
                    mask = self.get_roi_mask(roi_name=roi)
                    mask_group.create_dataset(roi, data=mask)

            #store patient id
            if 'patient_id' not in f.keys():
                f.attrs['patient_id'] = self.patient_id
        
        initial_guess, _ = self.grid_search()

        print(f'Initial Guess: {initial_guess}')
        res = self.minimizer(initial_guess)
        optimal_angles = res.x
        optimal_cost = res.fun
        return optimal_angles, optimal_cost
        

    def expensive_cost(self, polar, azimuthal):
        raise NotImplementedError

    def cost(self, x):
        raise NotImplementedError
    
    def set_azimuthal(self, azimuthal):
        raise NotImplementedError
    
    def set_polar(self, polar):
        raise NotImplementedError
    
    def set_gaze_angles(self, angles):
        raise NotImplementedError
    
    def calculate_dose(self):
        raise NotImplementedError
    
    def get_dose_at_volume(self, roi_name, volume):
        raise NotImplementedError
    
    def get_volume_at_dose(self, roi_name, dose):
        raise NotImplementedError
    
    def grid_search(self):
        raise NotImplementedError
    
    def get_metrics(self):
        raise NotImplementedError
    
    def cost_volume_term(self):
        raise NotImplementedError

    def fill_cache_from_csv(self):
        raise NotImplementedError
    
    def get_row_by_gaze_angle(self, gaze_angle):
        raise NotImplementedError
    
    def get_patient_id(self):
        raise NotImplementedError
    
    def get_dose(self):
        raise NotImplementedError
    
    def get_dvh(self, save_file_path):
        raise NotImplementedError

    def save_dose_distributions(self):
        raise NotImplementedError
    
    def get_current_wrapper(self, what):
        raise NotImplementedError

    def get_roi_mask(self, roi_name):
        raise NotImplementedError

    def get_roi_names(self):
        raise NotImplementedError
    
    def fill_cache_from_h5py(self):
        raise NotImplementedError
    
    def define_gaze_angle_grid(self):
        raise NotImplementedError

    def full_search(self):
        raise NotImplementedError
    
    

GazeOptimizer.full_search = full_search
GazeOptimizer.define_gaze_angle_grid = define_gaze_angle_grid
GazeOptimizer.fill_cache_from_h5py = fill_cache_from_h5py
GazeOptimizer.get_roi_mask = get_roi_mask
GazeOptimizer.get_roi_names = get_roi_names
GazeOptimizer.get_current_wrapper = get_current_wrapper
GazeOptimizer.save_dose_distributions = save_dose_distributions
GazeOptimizer.get_dvh = get_dvh
GazeOptimizer.get_dose = get_dose
GazeOptimizer.get_patient_id = get_patient_id
GazeOptimizer.cost_volume_term = cost_volume_term
GazeOptimizer.get_metrics = get_metrics
GazeOptimizer.expensive_cost = expensive_cost
GazeOptimizer.cost = cost
GazeOptimizer.grid_search = grid_search
GazeOptimizer.set_azimuthal = set_azimuthal
GazeOptimizer.set_polar = set_polar
GazeOptimizer.set_gaze_angles = set_gaze_angles
GazeOptimizer.calculate_dose = calculate_dose
GazeOptimizer.get_dose_at_volume = get_dose_at_volume
GazeOptimizer.get_volume_at_dose = get_volume_at_dose
GazeOptimizer.fill_cache_from_csv = fill_cache_from_csv
GazeOptimizer.get_row_by_gaze_angle = get_row_by_gaze_angle














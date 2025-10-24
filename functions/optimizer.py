from functions.RSAdapt.setDefaultPaths import setDefaultPaths
setDefaultPaths()
import sys
sys.path.append(r"C:\Program Files\RaySearch Laboratories\RayStation 2023B-R\ScriptClient")

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from functions.ray_wrappers import *
from functions.cost_functions import *
from functions.helpers import *
from functions.grid_search import *
from functions.data_storage import *
from functions.plotting import *

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
        extensive_path=None,
        optimal_path=None,
        clinical_path=None,
        use_precalculated=False,
        plot_path='../plots/'):
        
        self.columns = [('gaze_angles', 'polar'), ('gaze_angles', 'azimuthal')] + [('dvh_points', k) for k in weights.keys()] + [('dvh_points', 'dose_volume_penalty'), ('total_cost', 'cost')]
        self.cost_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(self.columns))
        self.patient_id = patient_id
        self.maxfev = maxfev
        self.delta_polar = delta_polar
        self.optimal_path = optimal_path
        self.clinical_path = clinical_path
        if save_path: self.save_path = save_path
        else: self.save_path = f"results/{self.patient_id}_eval_history.csv"
        self.prescribed_dose = prescribed_dose
        self.weights = weights
        self.extensive_path = extensive_path
        self.max_polar_deg = max_polar_deg
        self.bounds = [(0,max_polar_deg), (0, 360)]
        self.use_precalculated = use_precalculated

        if f'P{self.patient_id}' != self.get_patient_id():
            raise Exception(f'Check Patient ID: Raysearch: {self.get_patient_id()}, while here {self.patient_id}')

        if self.extensive_path:
            self.fill_cache_from_csv()
            self.cost_df = pd.concat([self.cost_df, pd.read_csv(self.extensive_path, header = [0,1])])
        else: self.cache = {}
    

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
        gaze_angles = define_gaze_angle_grid(
            delta_polar=self.delta_polar,
            max_polar_deg=self.max_polar_deg
        )

        initial_guess, _ = self.grid_search(gaze_angles)
        print('init guess')
        print(initial_guess)
        res = self.minimizer(initial_guess)
        optimal_angles = res.x
        self.cost_df.to_csv(f'save_eval/{self.patient_id}.csv')
        return optimal_angles
        

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
    
    def grid_search(self, all_angles):
        raise NotImplementedError
    
    def get_metrics(self):
        raise NotImplementedError
    
    def cost_volume_term(self):
        raise NotImplementedError

    def fill_cache_from_csv(self):
        raise NotImplementedError
    
    def get_row_by_gaze_angle(self, gaze_angle):
        raise NotImplementedError

    def single_gaze_plot(self, ax, metric, method):
        raise NotImplementedError

    def full_plot(self, method):
        raise NotImplementedError

    def find_optimum_for_metric(self, metric):
        raise NotImplementedError
    
    def find_optimum_cost(self):
        raise NotImplementedError

    def scatter_only_optima(self, ax):
        raise NotImplementedError
    
    def find_all_optima(self):
        raise NotImplementedError
    
    def get_patient_id(self):
        raise NotImplementedError
    

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
GazeOptimizer.single_gaze_plot = single_gaze_plot
GazeOptimizer.full_plot = full_plot
GazeOptimizer.find_optimum_for_metric = find_optimum_for_metric
GazeOptimizer.find_optimum_cost = find_optimum_cost
GazeOptimizer.scatter_only_optima = scatter_only_optima
GazeOptimizer.find_all_optima = find_all_optima












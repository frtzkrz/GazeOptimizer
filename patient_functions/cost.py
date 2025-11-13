
import numpy as np
from scipy.optimize import minimize_scalar
from patient_functions.dvh import cumulative_dvh, get_dose_at_volume, get_volume_at_dose, get_ray_dvh
import matplotlib.pyplot as plt

def cost_1_beam(self, gaze_angle, full_results=False):
    """
    Return cost for given gaze angle key given a set of weights
    weights: {{'roi_name': Macula, 'metric_type': 'D', 'value': 2, 'weight': 3}, ...} #D2_Macula weight 3
    """
    metric_term = self.gaze_angle_dvhs[gaze_angle].calculate_metric_term()

    volume_term = self.gaze_angle_dvhs[gaze_angle].calculate_volume_term()

    return metric_term + volume_term

def minimizer(self, dose_1, dose_2, metric_weights):
    cost_wrapper = lambda x: self.cost(weights=metric_weights, dose=x*dose_1 + (1-x)*dose_2, x=x)
    res = minimize_scalar(
        fun=cost_wrapper,
        bounds=(0,1),
        method='bounded',
    )
    return res

def combined_cost(self, dose_1, dose_2, weights):
    return


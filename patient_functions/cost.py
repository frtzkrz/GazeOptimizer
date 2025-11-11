
import numpy as np
from scipy.optimize import minimize_scalar
from patient_functions.dvh import cumulative_dvh, get_dose_at_volume, get_volume_at_dose, get_ray_dvh
import matplotlib.pyplot as plt

def cost(self, weights, total_dose):
    """
    Return cost for given gaze angle key given a set of weights
    weights: {{'roi': Macula, 'metric': 'D2', weight: 3}, ...}
    """
    ray_dvhs = get_ray_dvh()

    metric_term = 0
    volume_term = 0
    vols = np.arange(5, 101, 5)
    for i, weight in enumerate(weights):
        #unpack weight
        roi = weight['roi']
        metric = weight['metric']
        w = weight['weight']


        #get dose to roi
        dose_to_roi = total_dose[self.roi_mask_dict[roi]]

        #calculate dvh
        dose, volume = cumulative_dvh(dose_to_roi)

        #get metric value
        metric_value = self.get_metric(metric=metric, roi=roi, dose=dose_to_roi)
        print(f'Result: {roi, metric, metric_value}')
        metric_term += w * metric_value

    for roi in self.rois:
        w = next((item["weight"] for item in weights if item["roi"] == 'Retina'), 1)
        #volume term
        volume_term += w*np.sum([get_dose_at_volume(dose, vol)/100.0 for vol in vols])/np.size(vols)
    total_cost = metric_term + volume_term
    print(f'Volume term: {volume_term}')
    return total_cost


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
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from connect import *
import h5py

def get_metrics(self) -> Dict[str, float]:
    """
    Get relevant metrics of current Beam for calculating the cost
    Returns: {}
    """
    plan = get_current('Plan')
    metrics={}
    for w in self.weights.keys():
        what, where = w.split('_')
        val = int(what[1:])
        whatType = what[0]
        if whatType == 'D':
            metrics[w] = self.get_dose_at_volume(roi_name=where, relative_volumes=[val/100])[0]/100

        if whatType == 'V':
            metrics[w] = self.get_volume_at_dose(roi_name=where, dose_values=[val*100])[0]*100
    return metrics

def cost_volume_term(self) -> float:
    doses = np.arange(5, 101, 5)*self.prescribed_dose
    cost=0
    for w in self.weights.keys():
        _, where = w.split('_')
        cost += self.weights[w]*np.sum([self.get_volume_at_dose(roi_name=where, dose_values=[dose])*100 for dose in doses])/np.size(doses)     
    return cost


def expensive_cost(self, gaze_angle: Tuple[float, float]):
    """
    Cost Function that actually calculates the cost. Only to be used by cached_cost
    x = [polar, azimuthal]
    Saves full evaluation in h5 but returns only the cost.
    """
    with h5py.File(self.h5py_file_path, "a") as f:

        #Open/Create Group to store different gaze angles
        gaze_group = f.require_group('gaze_angles')

        #Create Group '(polar, azimuthal)'
        angle_group = gaze_group.require_group(f'{gaze_angle[0]}_{gaze_angle[1]}')
        angle_group.attrs['gaze_angle'] = gaze_angle

        #Calculate dose for current gaze angle
        self.set_gaze_angles(gaze_angle=gaze_angle)
        self.calculate_dose()

        roi_names = self.get_roi_names()

        #Save whole dose distribution and save to h5
        plan_dose = self.get_dose()
        angle_group.create_dataset("total_dose", data=plan_dose)

        #Get DVH for each ROI and save to h5
        for roi in roi_names:
            doses = self.get_dvh(roi)
            angle_group.create_dataset(roi, data=doses)


        metrics = self.get_metrics()
        volume_term = self.cost_volume_term()
        angle_group.attrs['volume_term'] = volume_term

        angle_group.attrs.update(metrics)

        weights_arr = np.array([float(i) for i in self.weights.values()])
        metrics_arr = np.array([float(i) for i in metrics.values()])
        total_cost = weights_arr@metrics_arr + volume_term

        angle_group.attrs['total_cost'] = total_cost

    
    return total_cost


def cost(self, gaze_angle: Tuple[float, float]):
    """
    Wrapper for optimizer: x = [polar, azimuthal]
    Checks in self.cost_df if cost was already evaulated
    Saves full evaluation in eval_df but returns only the cost.

    gaze_angle: 
    """
    
    gaze_angle = np.round(gaze_angle, 1)
    key = tuple(gaze_angle)

    if key in self.cache:
        return self.cache[key]

    else:
        total_cost = self.expensive_cost(gaze_angle=gaze_angle)

        # Cache the cost
        self.cache[key] = total_cost

        return total_cost


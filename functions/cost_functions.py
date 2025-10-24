from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from connect import *

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
            metrics[w] = self.get_dose_at_volume(roi_name=where, relative_volumes=[val/100])[0]

        if whatType == 'V':
            metrics[w] = self.get_volume_at_dose(roi_name=where, dose_values=[val*100])[0]
    return metrics

def cost_volume_term(self) -> float:
    doses = np.arange(5, 101, 5)*self.prescribed_dose
    cost=0
    for w in self.weights.keys():
        _, where = w.split('_')
        cost += self.weights[w]*np.sum([self.get_volume_at_dose(where, [dose]) for dose in doses])     
    return cost


def expensive_cost(self, gaze_angle: Tuple[float, float]):
    """
    Cost Function that actually calculates the cost. Only to be used by cached_cost
    x = [polar, azimuthal]
    Saves full evaluation in eval_df but returns only the cost.
    """
    self.set_gaze_angles(gaze_angle=gaze_angle)
    self.calculate_dose()
    metrics = self.get_metrics()
    volume_term = self.cost_volume_term()
    weights_arr = np.array([float(i) for i in self.weights.values()])
    metrics_arr = np.array([float(i) for i in metrics.values()])
    total_cost = weights_arr@metrics_arr + volume_term

    #Add results to self.cost_df
    new_row = {('dvh_points', metric): metrics[metric] for metric in metrics}
    new_row[('gaze_angles', 'polar')] = gaze_angle[0]
    new_row[('gaze_angles', 'azimuthal')] = gaze_angle[1]
    new_row[('dvh_points', 'dose_volume_penalty')] = volume_term
    new_row[('total_cost', 'cost')] =  total_cost
    new_row_df = pd.DataFrame(new_row, index=[0])
    self.cost_df = pd.concat([self.cost_df, new_row_df])
    self.cost_df.to_csv(self.save_path)
    
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
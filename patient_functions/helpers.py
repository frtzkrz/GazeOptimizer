from __future__ import annotations 
from typing import List, Tuple
import numpy as np

def cumulative_dvh(dose, frac, voxel_vol, bins=1000):
    assert dose.shape == frac.shape
    v = frac * voxel_vol  # volume contributed by each voxel
    dmin = 0
    dmax = float(dose.max())
    
    total_vol = np.sum(v)

    # Make bins (edges)
    edges = np.linspace(dmin, dmax, bins + 1)
    # differential DVH: sum volumes per dose bin
    bin_idx = np.searchsorted(edges, dose, side='right') - 1
    # clamp indices
    bin_idx = np.clip(bin_idx, 0, bins-1)
    diff = np.bincount(bin_idx, weights=v, minlength=bins)  # volumes per bin
 
    # cumulative DVH (volume >= D) — compute from the top
    # create center points for bins (optional)
    dose = 0.5 * (edges[:-1] + edges[1:])
    # cumulative from high dose to low dose:
    vol = np.cumsum(diff[::-1])[::-1]/total_vol*100
 
    return dose, vol

def get_ray_dvh(path='results/P23336_ray_dvhs.txt'):
    dvh_data = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        current_roi = None
        for line in lines:
            line = line.strip()
            if line.startswith('#RoiName'):
                current_roi = line.split(':')[1].strip()
                dvh_data[current_roi] = {'dose': [], 'volume': []}
            elif current_roi and line and not line.startswith('#'):
                dose_val, vol_val = map(float, line.split())
                dvh_data[current_roi]['dose'].append(dose_val)
                dvh_data[current_roi]['volume'].append(vol_val)
    return dvh_data

# Print iterations progress
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({iteration}/{total})', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_angle_from_key(
    angle_key: str, 
    azimuthal_as_radian: bool=True
    ) -> Tuple[float]:
    """
    Extract polar and azimuthal angles from a treatment plan's angle key.
    Parameters:
    angle_key (str): The angle key string (e.g., '25.0_125.5').
    theta (bool): If True, return azimuthal angle in radians. If False, return in degrees.
    Returns:
    tuple[float]: A list containing polar and azimuthal angles.
    """
    polar = float(plan.angle_key.split('_')[0][1:])
    azimuthal = float(plan.angle_key.split('_')[1][:-1])
    if azimuthal_as_radian: return polar, np.deg2rad(azimuthal)
    else: return polar, azimuthal

def get_angles_from_keys(
    angle_keys: List[str], 
    azimuthal_as_radian: bool=True
    ) -> Tuple[List[float]]:

    """
    Extract polar and azimuthal angles from a list of treatment plan angle keys.
    Parameters:
    angle_keys (List[str]): List of angle key strings (e.g., ['25.0_125.5', '30.0_130.0']).
    theta (bool): If True, return azimuthal angles in radians. If False, return in degrees.
    Returns:
    tuple[List[float]]: Two lists containing polar and azimuthal angles.
    """
    polars = []
    azimuthals = []
    for angle_key in angle_keys:
        polar, azimuthal = get_angle_from_key(angle_key, azimuthal_as_radian=azimuthal_as_radian)
        polars.append(polar)
        azimuthals.append(azimuthal)
    return polars, azimuthals

def find_best_plan(plans: List[TreatmentPlan]) -> TreatmentPlan:
    """
    Find the treatment plan with the lowest cost.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to evaluate.
    Returns:
    TreatmentPlan: The plan with the lowest cost.
    """
    best_plan = None
    lowest_cost = float('inf')
    for plan in plans:
        cost = plan.calculate_cost()
        if cost < lowest_cost:
            lowest_cost = cost
            best_plan = plan
    return best_plan

def roi_in_metrics(
    roi: str, 
    weights: Weights
    ) -> bool:
    """
    Check if there is a metric corresponding to the given ROI.
    Parameters:
    roi (str): The region of interest to check.
    Returns:
    bool: True if there is a metric for the ROI, False otherwise."""
    for weight in weights:
        if roi in weight.metric.roi:
            return weight
    return False

def find_metric_for_roi(roi, weights):
    #find metric corresponding to roi
    metric = roi_in_metrics(roi=roi, weights=weights)
    if not metric:
        raise ValueError(f'ROI {roi} not found in weights.')
    else: 
        return metric
    
def filter_plans(
    plans: List[TreatmentPlan], 
    filter_dict: dict
    ) -> List[TreatmentPlan]:
    """
    Filter treatment plans based on a specified metric threshold.
    Parameters:
    plans (List[TreatmentPlan]): List of TreatmentPlan objects to filter.
    filter_dict (dict): Dictionary containing filter information (e.g., D50_Macula < 10 Gy {'metric': 'D50', 'roi': 'Macula', 'max': 10}).
    Returns:
    List[TreatmentPlan]: Filtered list of TreatmentPlan objects.
    """
    filtered_plans = [plan for plan in plans if plan.dvhs[filter_dict['roi'].get_metric_value(metric) < filter_dict['max']]]


    
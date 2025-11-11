import numpy as np

def find_gaze_angle_smaller_dvol(self, roi, vol, max_dose, filtered_gaze_angle_keys=None):
    """
    Find all gaze angles that have D_vol smaller than max_dose Gy for given roi

    if filtered_gaze_angle_keys is given, only search within those keys
    roi: str
    vol: 0<vol<100
    maxdose: float/int in Gy
    """
    #find bin index that corresponds to volume of interest
    vol_idx = int(np.round(self.num_dvh_bins*vol/100))

    #argmin scaled to accomodate different num_dvh_bins selections
    if filtered_gaze_angle_keys is not None:
        dvhs = {k: dvhs[k] for k in filtered_gaze_angle_keys}
    else: dvhs = self.dvh_dict[roi]

    #find gaze angles for which D_vol < max_dose
    filtered_gaze_angle_keys = [gaze_angle_key for gaze_angle_key in dvhs if dvhs[gaze_angle_key][vol_idx]<max_dose*100] #mult by 100 bc dose is stored in cGy
    return filtered_gaze_angle_keys

def find_gaze_angle_smaller_vdose(self, roi, dose, max_vol, filtered_gaze_angle_keys=None):
    """
    Find all gaze angles that have V_dose smaller than max_vol Gy for given roi

    if filtered_gaze_angle_keys is given, only search within those keys
    roi: str
    max_vol: 0<vol<100
    dose: float/int in Gy
    """

    #argmin scaled to accomodate different num_dvh_bins selections
    if filtered_gaze_angle_keys is not None:
        dvhs = {k: self.dvh_dict[roi][k] for k in filtered_gaze_angle_keys}
    else: dvhs = self.dvh_dict[roi]


    #find gaze angles for which V_dose < max_vol
    #dose scaled by 100 for cGy>Gy

    filtered_gaze_angle_keys = [gaze_angle_key for gaze_angle_key in dvhs if np.round(np.abs(dvhs[gaze_angle_key] - dose*100).argmin()/100*self.num_dvh_bins)<max_vol]
    return filtered_gaze_angle_keys

def find_new_optimal_gaze_angles(self, filtered_gaze_angle_keys):
    """
    Find new optimal gaze angles and costs from filtered gaze angle keys
    """
    filtered_costs = [self.costs_dict[k] for k in filtered_gaze_angle_keys]
    if len(filtered_costs)==0:
        print("No gaze angles found after filtering!")
        return None, None

    idx_min_filtered = np.argmin(filtered_costs)
    optimal_gaze_angle_key = filtered_gaze_angle_keys[idx_min_filtered]
    optimal_cost = filtered_costs[idx_min_filtered]
    return {'New Optimum': optimal_gaze_angle_key}
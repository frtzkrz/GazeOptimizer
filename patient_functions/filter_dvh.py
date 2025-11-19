import numpy as np

def filter_gaze_angle_dvhs(dvhs, roi, vol, max_dose):
    return [dvh for dvh in dvhs if dvhs[roi].get_dose_at_volume(vol) < max_dose]


def find_gaze_angle_smaller_dvol(self, roi, vol, max_dose, filtered_gaze_angle_keys=None):
    """
    Find all gaze angles that have D_vol smaller than max_dose Gy for given roi

    if filtered_gaze_angle_keys is given, only search within those keys
    roi: str
    vol: 0<vol<100
    maxdose: float/int in Gy
    """
    gaze_angle_selection = filtered_gaze_angle_keys if filtered_gaze_angle_keys is not None else self.gaze_angle_keys
    filtered_gaze_angle_keys = [gaze_angle_key for gaze_angle_key in gaze_angle_selection if self.gaze_angle_dvhs[gaze_angle_key].roi_dvhs[roi].get_dose_at_volume(vol) < max_dose]
    return filtered_gaze_angle_keys


def find_gaze_angle_smaller_vdose(self, roi, dose, max_vol, filtered_gaze_angle_keys=None):
    """
    Find all gaze angles that have V_dose smaller than max_vol Gy for given roi

    if filtered_gaze_angle_keys is given, only search within those keys
    roi: str
    max_vol: 0<vol<100
    dose: float/int in Gy
    """
    gaze_angle_selection = filtered_gaze_angle_keys if filtered_gaze_angle_keys is not None else self.gaze_angle_keys
    filtered_gaze_angle_keys = [gaze_angle_key for gaze_angle_key in gaze_angle_selection if self.gaze_angle_dvhs[gaze_angle_key].roi_dvhs[roi].get_volume_at_dose(dose) < max_vol]
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

def apply_dvh_filters(self, dvh_filters):
    #filter = {'roi_name': 'Macula', 'filter_type': 'D', 'value': 50, 'max_val': 30} <- D50_Macula < 30Gy
    current_keys = self.gaze_angle_keys
    for dvh_filter in dvh_filters:
        if dvh_filter['filter_type'] == 'D':
            current_keys = self.find_gaze_angle_smaller_dvol(roi=dvh_filter['roi_name'], vol=dvh_filter['value'], max_dose=dvh_filter['max_val'], filtered_gaze_angle_keys=current_keys)
        elif dvh_filter['filter_type'] == 'V':
            current_keys = self.find_gaze_angle_smaller_vdose(roi=dvh_filter['roi_name'], dose=dvh_filter['value'], max_vol=dvh_filter['max_val'], filtered_gaze_angle_keys=current_keys)
        if len(current_keys) == 0:
            print('No gaze angle exists.')
            print(f'(Stopped at {dvh_filter})')
    return current_keys
def find_gaze_angle_smaller_dvol(patient, roi, vol, max_dose, gaze_angle_keys=None):
    """
    Find all gaze angles that have D_vol smaller than max_dose Gy for given roi. 
    If filtered_gaze_angle_keys is given, only search within those keys

    patient: instance of Patient
    roi: Region of Interest
    vol: 0<vol<100 (%)
    maxdose: float/int in Gy
    """
    return
from patient_functions.roi_dvh import RoiDVH

class GazeAngleDVHs:
    def __init__(
        self,
        patient,
        angle_key,
        dose,
        angle_key_2=None,
        weight=None,
    ):  
        self.patient = patient
        self.angle_key = angle_key
        self.weight = weight
        if angle_key_2 is not None and weight is not None:
            self.weight = weight
            self.angle_key_2 = angle_key_2
        
        self.roi_dvhs = {}
        for roi_name in patient.roi_names:
            self.roi_dvhs[roi_name] = RoiDVH(patient, roi_name, dose)
    
    def calculate_cost(self):
        metric_term = self.calculate_metric_term()
        volume_term = self.calculate_volume_term()
        self.cost = metric_term + volume_term
        return self.cost

    def calculate_volume_term(self):
        volume_term = 0
        for roi_name in self.patient.roi_names:
            w = next((item["weight"] for item in self.patient.weights if item["roi_name"] == roi_name), 1)
            x = self.roi_dvhs[roi_name].get_dvh_auc()/100
            volume_term += w * x
        return volume_term
    
    def calculate_metric_term(self):
        metric_term = 0
        for i, weight in enumerate(self.patient.weights):
            #print(weight)
            if weight['metric_type'] == 'D':
                metric_value = self.roi_dvhs[weight['roi_name']].get_dose_at_volume(volume=weight['value'])
            
            else: 
                metric_value = self.roi_dvhs[weight['roi_name']].get_volume_at_dose(dose=weight['value'])

            metric_term += weight['weight'] * metric_value
        return metric_term
from patient_functions.dvh import cumulative_dvh
import numpy as np

class RoiDVH:
    def __init__(
        self,
        patient,
        roi_name,
        dose
        ):
        self.patient_id = patient.patient_id
        self.roi_name = roi_name
        self.dose, self.volume = cumulative_dvh(
            dose=dose[patient.roi_masks[self.roi_name]], 
            frac=patient.roi_relative_values[self.roi_name], 
            voxel_vol = patient.voxel_vol, 
            bins=patient.num_dvh_bins)

    def plot(self, ax, plot_args={}):
        ax.plot(self.dose, self.volume, **plot_args)
    
    def get_dvh(self):
        return self.dose, self.volume
    
    def get_volume_at_dose(self, dose):
        """
        #Return Vx: the volume (%) receiving at least dose.
        #dvh_dose, dvh_volume can be ascending or descending; handles both.
        """
        # Ensure numpy arrays
        dvh_dose = np.asarray(self.dose)
        dvh_volume = np.asarray(self.volume)
        
        # Make sure dose is ascending for interpolation
        if dvh_dose[0] > dvh_dose[-1]:
            dvh_dose = dvh_dose[::-1]
            dvh_volume = dvh_volume[::-1]
        
        # Interpolate volume at dose x
        return np.interp(dose, dvh_dose, dvh_volume)

    def get_dose_at_volume(self, volume):
        """
        Return Dx: the dose corresponding to volume x.
        - x can be scalar or array (volume units).
        - If clip=True (default), x outside the range of `volume` is clipped to min/max.
        If clip=False, np.interp will extrapolate using end values (which is usually undesirable).
        Assumes `volume` is cumulative DVH (monotonic, typically decreasing with dose).
        """
        dvh_dose = np.asarray(self.dose)
        dvh_volume = np.asarray(self.volume)

        # np.interp requires the xp (here: volume) to be ascending.
        # If volume is descending, reverse both arrays to make volume ascending.
        if dvh_volume[0] > dvh_volume[-1]:
            dvh_dose = dvh_dose[::-1]
            dvh_volume = dvh_volume[::-1]
        volume = np.asarray(volume)
        volume = np.clip(volume, dvh_volume.min(), dvh_volume.max())
        # interpolate dose as a function of volume (xp=volume, fp=dose)
        return np.interp(volume, dvh_volume, dvh_dose)
    
    def get_dvh_auc(self):
        return np.trapz(y=self.volume, x=self.dose)
    
    def get_metric_value(self, metric):
        if metric['metric_type'] == 'D':
            return self.get_dose_at_volume(metric['value'])
        else: return self.get_volume_at_dose(metric['value'])
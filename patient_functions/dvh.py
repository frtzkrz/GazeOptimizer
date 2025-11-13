import numpy as np
import h5py
import matplotlib.pyplot as plt

def test_gaze_combination(self, gaze_angle_keys, total_dose, weight1):
    """
    Return DVH for given roi and dose distribution
    """

    y = np.linspace(0, 100, 100)
    for roi in self.rois:
        plt.figure()
        
        dvh_opt = self.dvh_dict[roi][self.optimal_angle_key]
        plt.plot(dvh_opt,y, '--', label='Optimum single field')

        dvh_new = self.dvh_dict[roi][gaze_angle_keys[0]]
        plt.plot(dvh_new, y, '--', label=f'F1')

        dvh_new = self.dvh_dict[roi][gaze_angle_keys[1]]
        plt.plot(dvh_new, y, '--', label=f'F2')

        dose, volume = cumulative_dvh(total_dose[self.roi_mask_dict[roi]], num_points=self.num_dvh_bins)
        plt.title(f'Combined DVH for ROI: {roi}')
        plt.plot(dose, volume, label=f'{np.round(weight1,2)}*F1 + {1-np.round(weight1,2)}*F2')
        plt.grid()
        plt.legend()
        plt.show()
    return 

def get_metric(self, metric, roi, dose):
    """
    Get metric value from dose array.

    Parameters
    ----------
    metric : str
        Metric type ('Dxx' or 'Vxx').
    roi : str
        Region of interest name.
    total_dose : array-like
        Dose values (in cGy) for all voxels of the organ.

    Returns
    -------
    float
        Metric value (dose in cGy for Dxx, volume fraction for Vxx).
    """

    if metric.startswith('D'):
        volume = float(metric[1:])
        print(roi, metric, volume)
        return get_dose_at_volume(dose, volume)/100.0
    elif metric.startswith('V'):
        dose_value = float(metric[1:])
        print(roi, metric, dose_value)
        return get_volume_at_dose(dose, dose_value)
    else:
        raise ValueError("Metric must start with 'D' or 'V'.")



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



#NOT UPDATED ANYMORE; USE METHOD OF ROIDVH, delete eventually
def get_volume_at_dose(dvh_dose, dvh_volume, dose):
    """
    Return Vx: the volume (%) receiving at least dose.
    dvh_dose, dvh_volume can be ascending or descending; handles both.
    """
    # Ensure numpy arrays
    dvh_dose = np.asarray(dvh_dose)
    dvh_volume = np.asarray(dvh_volume)
    
    # Make sure dose is ascending for interpolation
    if dvh_dose[0] > dvh_dose[-1]:
        dvh_dose = dvh_dose[::-1]
        dvh_volume = dvh_volume[::-1]
    
    # Interpolate volume at dose x
    return np.interp(dose, dvh_dose, dvh_volume)


#NOT UPDATED ANYMORE; USE METHOD OF ROIDVH, delete eventually
def get_dose_at_volume(dvh_dose, dvh_volume, volume, clip=True):
    """
    Return Dx: the dose corresponding to volume x.
    - x can be scalar or array (volume units).
    - If clip=True (default), x outside the range of `volume` is clipped to min/max.
      If clip=False, np.interp will extrapolate using end values (which is usually undesirable).
    Assumes `volume` is cumulative DVH (monotonic, typically decreasing with dose).
    """
    dvh_dose = np.asarray(dvh_dose)
    dvh_volume = np.asarray(dvh_volume)

    # np.interp requires the xp (here: volume) to be ascending.
    # If volume is descending, reverse both arrays to make volume ascending.
    if dvh_volume[0] > dvh_volume[-1]:
        dvh_dose = dvh_dose[::-1]
        dvh_volume = dvh_volume[::-1]
    volume = np.asarray(volume)
    volume = np.clip(volume, dvh_volume.min(), dvh_volume.max())
    # interpolate dose as a function of volume (xp=volume, fp=dose)
    return np.interp(volume, dvh_volume, dvh_dose)

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


def compare_dvhs(self, ray_path, plot_folder):
    """
    Compares selfcalculated 
    """
    ray_dvhs = get_ray_dvh(path=ray_path)
    plt.figure()
    for i, roi_name in enumerate(self.roi_names):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        dose_ray = np.array(ray_dvhs[roi_name]['dose'])*100
        vol_ray = ray_dvhs[roi_name]['volume']

        my_dose, my_vol = self.gaze_angle_dvhs['(0, 0)'].roi_dvhs[roi_name].get_dvh()

        plt.plot(dose_ray/100, vol_ray, linestyle='--', color=color)
        plt.plot(my_dose, my_vol, color=color)
    
    plt.plot([], [], 'k--', label='RayStation DVH')
    plt.plot([], [], 'k-', label='Calculated DVH')
    plt.title('DVH Comparison')
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(f'{plot_folder}/{self.patient_id}.png', dpi=200)

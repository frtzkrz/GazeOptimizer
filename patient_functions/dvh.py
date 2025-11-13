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

#def cumulative_dvh(total_dose, num_points=1000):
    """
    Compute cumulative dose-volume histogram (DVH).

    Parameters
    ----------
    dose_array : array-like
        Dose values (in cGy) for all voxels of the organ.
    num_points : int
        Number of dose bins (default: 100).

    Returns
    -------
    dose_bins : np.ndarray
        Dose values corresponding to the DVH curve (in cGy).
    dvh : np.ndarray
        Cumulative volume fraction (0–100) for each dose bin.
    """
    """total_dose = np.asarray(total_dose)
    
    # Define equally spaced dose bins from 0 to max dose
    dose = np.linspace(0, total_dose.max(), int(num_points))
    
    # Sort dose values once for efficiency
    sorted_dose = np.sort(total_dose)
    n = len(sorted_dose)

    # For each bin, find how many voxels ≥ dose (using searchsorted for speed)
    counts = n - np.searchsorted(sorted_dose, dose, side='left')

    # Normalize to get %
    vol = 100*(counts / n)

    number_roi_voxels = np.size(total_dose)
    

    return dose, vol"""


def cumulative_dvh(dose, frac, voxel_vol, bins=1000, dose_range=(0,6000)):
    assert dose.shape == frac.shape
    v = frac * voxel_vol  # volume contributed by each voxel
 
    if dose_range is None:
        dmin, dmax = float(dose.min()), float(dose.max())
    else:
        dmin, dmax = dose_range
    
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


def get_dose_at_volume(dose_array, volume):
    """
    Get dose at specified volume from DVH.

    Parameters
    ----------
    dvh : np.ndarray
        dose values (in cGy)
    volume : float
        Volume percentage (0-100) at which to find the dose.

    Returns
    -------
    dose : float
        Dose at volume in cGy
    """
    # Find index where dvh is just above the specified volume

    dose = dose_array[int(volume * (len(dose_array)-1) / 100.0)]
    return dose

def get_volume_at_dose(dose_array, dose):
    """
    Get volume at specified dose from DVH.

    Parameters
    ----------
    dvh : np.ndarray
        Cumulative volume fraction (0–1) for each dose bin.
    dose : float
        Dose value at which to find the volume.

    Returns
    -------
    volume : float
        Volume in % at dose
    """
    # Find index where dvh is just below the specified dose
    n_bins = len(dose_array)
    volume_index = np.where(dose_array >= dose)[0][-1]
    volume = np.linspace(0, 100, n_bins)[volume_index]
    return volume

def get_ray_dvh(path='results/P23336_ray_dvhs.txt'):
    dvh_data = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        current_roi = None
        for line in lines:
            line = line.strip()
            if line.startswith('#RoiName'):
                current_roi = line.split(':')[1].strip()
                print(current_roi)
                dvh_data[current_roi] = {'dose': [], 'volume': []}
            elif current_roi and line and not line.startswith('#'):
                dose_val, vol_val = map(float, line.split())
                dvh_data[current_roi]['dose'].append(dose_val)
                dvh_data[current_roi]['volume'].append(vol_val)
    return dvh_data


def compare_dvhs(self, ray_path, total_dose):
    ray_dvhs = get_ray_dvh(path=ray_path)
    plt.figure()
    for i, roi in enumerate(self.rois):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        dose_ray = np.array(ray_dvhs[roi]['dose'])*100
        vol_ray = ray_dvhs[roi]['volume']

        my_dose, my_vol = cumulative_dvh(total_dose[self.roi_mask_dict[roi]])

        
        plt.plot(dose_ray, vol_ray, linestyle='--', color=color)
        plt.plot(my_dose, my_vol, color=color)
    
    plt.plot([], [], 'k--', label='RayStation DVH')
    plt.plot([], [], 'k-', label='Calculated DVH')
    plt.title('DVH Comparison (-- RayStation)')
    plt.xlabel('Dose (cGy)')
    plt.ylabel('Volume (%)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

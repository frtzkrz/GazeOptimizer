from optimizer_functions.RSAdapt.setDefaultPaths import setDefaultPaths
setDefaultPaths()
import sys
sys.path.append(r"C:\Program Files\RaySearch Laboratories\RayStation 2023B-R\ScriptClient")

from connect import *
from patient_functions.dvh import cumulative_dvh, get_ray_dvh
import matplotlib.pyplot as plt

import numpy as np
 
 
# Example usage:
# dose_arr, frac_arr are flattened numpy arrays
voxel_volume = .000064  # example: mm->cc as appropriate
# edges, centers, diff, cum = compute_dvh_from_fractions(dose_arr, frac_arr, voxel_volume, bins=200)
beam_set = get_current('BeamSet')
sub_structure_set = beam_set.DependentSubStructureSet
roi_names = [r.OfRoi.Name for r in sub_structure_set.RoiStructures if r.PrimaryShape is not None]
roi_names = [n for n in roi_names if n not in ["External"] and 'clip' not in n.lower()]

ray_dvhs = get_ray_dvh(path=f'results/P23336_ray_dvhs.txt')

plan = get_current('Plan')
plan_dose = plan.TreatmentCourse.TotalDose.DoseValues.DoseData.flatten()

for i, roi in enumerate(roi_names):
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]



    dose_ray = np.array(ray_dvhs[roi]['dose'])*100
    vol_ray = ray_dvhs[roi]['volume']


    dgr = plan.TreatmentCourse.TotalDose.GetDoseGridRoi(RoiName = roi)
    roi_mask = dgr.RoiVolumeDistribution.VoxelIndices.flatten()
    rel_vols = dgr.RoiVolumeDistribution.RelativeVolumes
    roi_dose = plan_dose[roi_mask]
    my_dose, my_vol = cumulative_dvh(dose=roi_dose, frac=rel_vols, voxel_vol=0.000064)


    plt.plot(dose_ray, vol_ray, linestyle='--', color=color)
    plt.plot(my_dose, my_vol, color=color)

plt.plot([], [], 'k--', label='RayStation DVH')
plt.plot([], [], 'k-', label='Calculated DVH')
plt.title('DVH Comparison (-- RayStation)')
plt.xlabel('Dose (cGy)')
plt.ylabel('Volume (%)')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('plots/ray_dvhs_comparison/P23336_new.png', dpi=200)
plt.show()
#Connection to RayStation
from RSAdapt.setDefaultPaths import setDefaultPaths
setDefaultPaths()
import sys
sys.path.append(r"C:\Program Files\RaySearch Laboratories\RayStation 2023B-R\ScriptClient")

from connect import *

import time
import h5py
import numpy as np
import math

def define_gaze_angle_grid(
        delta_polar,
        max_polar_deg,
    ):

    """
    Creates list of dict of gaze angles spreading the plane
    delta_polar: determines the spacing between points
    """
    angles = [(0, 0)]
    polars = np.linspace(0, max_polar_deg, int(np.ceil(max_polar_deg/delta_polar)))
    for p in polars:
        if not p==0:
            azimuthals = np.round(np.linspace(0, 360, int(np.ceil(2*np.pi*p/delta_polar)), endpoint=False),1)
            for a in azimuthals:
                angles.append((p, a))
    return angles


def grid_search(gaze_angles, h5py_path, patient_id):
    plan = get_current('Plan')
    beam_set = get_current('BeamSet')
    field = beam_set.Beams[0]

    with h5py.File(h5py_path, "a") as f:
        f.attrs['patient_id'] = patient_id
        sub_structure_set = beam_set.DependentSubStructureSet
        roi_names = [r.OfRoi.Name for r in sub_structure_set.RoiStructures if r.PrimaryShape is not None]
        roi_names = [n for n in roi_names if n not in ["External"] and 'clip' not in n.lower() and 'ep' not in n.lower()]
        f.attrs['roi_names'] = roi_names
        f.attrs['voxel_volume'] = math.prod(plan.TreatmentCourse.TotalDose.InDoseGrid.VoxelSize.values())
        

        for roi_name in roi_names:

            dgr = plan.TreatmentCourse.TotalDose.GetDoseGridRoi(RoiName = roi_name)

            #extract mask for roi
            roi_mask = dgr.RoiVolumeDistribution.VoxelIndices.flatten()
            f.create_dataset(f'{roi_name}_mask', data=roi_mask)

            #extract relative volumes of voxels belonging to roi
            roi_rel_vols = dgr.RoiVolumeDistribution.RelativeVolumes
            f.create_dataset(f'{roi_name}_relative_volumes', data=roi_rel_vols)

        for angle in gaze_angles:

            #set gaze angles
            field.GazeAngles.PolarGazeAngle = angle[0]
            field.GazeAngles.AzimuthalGazeAngle = angle[1]

            #calculate dose
            beam_set.TreatAndProtect(ShowProgress=False)
            beam_set.ComputeDose(ComputeBeamDoses=True, DoseAlgorithm="IonPencilBeam", ForceRecompute=False, RunEntryValidation=True)

            plan_dose = plan.TreatmentCourse.TotalDose.DoseValues.DoseData.flatten()/100
            f.create_dataset(str(angle), data=plan_dose)

def main():
    patient_id = 'P23336'
    
    start_time = time.time()
    gaze_angles = [(0,0)]+[(25, a) for a in np.arange(0, 360, 45)]
    grid_search(
        gaze_angles=gaze_angles,
        h5py_path=f'test/{patient_id}_9_angles.h5',
        patient_id=patient_id,
    )


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
    
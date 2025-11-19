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

import argparse


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

        total = len(gaze_angles)
        for i, angle in enumerate(gaze_angles):
            print(f"\rExtracting data from RayStation |{'x' * int(100 * (i+1) // total) + '-' * (100 - int(100 * i // total))}| ({i+1}/{total})", end = "\r")

            #set gaze angles
            field.GazeAngles.PolarGazeAngle = angle[0]
            field.GazeAngles.AzimuthalGazeAngle = angle[1]

            #calculate dose
            beam_set.TreatAndProtect(ShowProgress=False)
            beam_set.ComputeDose(ComputeBeamDoses=True, DoseAlgorithm="IonPencilBeam", ForceRecompute=False, RunEntryValidation=True)

            plan_dose = plan.TreatmentCourse.TotalDose.DoseValues.DoseData.flatten()/100
            f.create_dataset(str(angle), data=plan_dose)
        print("\nData extraction complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Extract data from RayStation for different gaze angles.")

    parser.add_argument("patient_id", type=str, help="Patient ID", default="23129")
    parser.add_argument("h5py_path", type=str, help="Path to save the h5py file")
    parser.add_argument("delta_azimuthal", type=float, help="Azimutahl angle delta. Only searches at (0, 0) and (25, x).")
    parser.add_argument("--delta_polar", type=float, default=None, help="Delta polar angle for grid search. If None, only (0, 0) and (25, x) are evaluated.")
    parser.add_argument("--max_polar_deg", type=float, default=25, help="Maximum polar angle for grid search. Default is 25 degrees.")

    args = parser.parse_args()
    #h5py_file_path = f"{args.h5py_path}/{args.patient_id}_delta_{args.delta_azimuthal}.h5"
    
    patient = get_current("Patient")
    ray_patient_id = patient.PatientID

    if args.patient_id != ray_patient_id:
            raise Exception(f'Check Patient ID: Raysearch: {ray_patient_id}, while here {args.patient_id}')
    


    h5py_path = f"{args.h5py_path}/{args.patient_id}_delta_{int(args.delta_azimuthal)}.h5"

    start_time = time.time()
    if args.delta_polar is not None:
        gaze_angles = define_gaze_angle_grid(
            delta_polar=args.delta_polar,
            max_polar_deg=args.max_polar_deg,
        )
    else:
        gaze_angles = [(0,0)]+[(25, a) for a in np.arange(0, 360, args.delta_azimuthal)]

    
    grid_search(
        gaze_angles=gaze_angles,
        h5py_path=h5py_path,
        patient_id=args.patient_id,
    )


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
    
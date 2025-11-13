import numpy as np
from connect import *
from typing import List
import pandas as pd
import math
import json

def set_azimuthal(self, azimuthal):
    """
    Set Azimuthal angle of specific field to angle (deg).
    field: IonBeam
    """
    if azimuthal>360 or azimuthal<0:
        raise ValueError("Azimuthal angle must be between 0 and 360 deg")
    beam_set = get_current("BeamSet")
    field = beam_set.Beams[0]
    field.GazeAngles.AzimuthalGazeAngle = azimuthal

def set_polar(self, polar):
    """
    Set polar angle (0<angle<25)
    """
    if polar>25 or polar<0:
        raise ValueError("Polar angle must be between 0 and 25 deg")
    beam_set = get_current("BeamSet")
    field = beam_set.Beams[0]
    field.GazeAngles.PolarGazeAngle = polar

def set_gaze_angles(self, gaze_angle):
    """
    Set polar and azimuthal angles
    angles: (polar, azimuthal)
    """
    self.set_polar(gaze_angle[0])
    self.set_azimuthal(gaze_angle[1])
        

def calculate_dose(self):
    """
    Wrapper for RaySearch.
    Runs TreatAndProtect and ComputeDose
    """
    beam_set = get_current("BeamSet")
    beam_set.TreatAndProtect(ShowProgress=False)
    beam_set.ComputeDose(ComputeBeamDoses=True, DoseAlgorithm="IonPencilBeam", ForceRecompute=False, RunEntryValidation=True)

def get_dose_at_volume(
    self,
    roi_name:str, 
    relative_volumes: List[float],
) -> List[float]:
    """
    Wrapper for getting D_v
    RelativeVolumes: [.99, .95, ...] (0 < RelativeVolumes < 1)
    """
    plan = get_current("Plan")
    return plan.TreatmentCourse.TotalDose.GetDoseAtRelativeVolumes(RoiName=roi_name, RelativeVolumes=relative_volumes)

def get_volume_at_dose(
        self,
        roi_name:str, 
        dose_values: List[float],
        ) -> List[float]:
    """
    Wrapper for getting V_d
    DoseValues: [100, 500, ...] in cGy
    """
    plan = get_current("Plan")
    return plan.TreatmentCourse.TotalDose.GetRelativeVolumeAtDoseValues(RoiName=roi_name, DoseValues=dose_values)

def get_patient_id(self):
    patient = get_current("Patient")
    return patient.PatientID

def load_patient(patient_id):
    patient_db = get_current("PatientDB")
    patient = patient_db.QueryPatientInfo(Filter = {'PatientID': patient_id} )
    print(patient)
    patient_db.LoadPatient(PatientInfo=patient, AllowPatientUpgrade=True)
    return patient_db

def get_dose(self):
    plan = get_current('Plan')
    plan_dose = plan.TreatmentCourse.TotalDose.DoseValues.DoseData
    return plan_dose.flatten()


def get_dvh(self, roi):
    """ Compute cumulative DVH"""
    rel_vols = np.linspace(0,1,self.num_dvh_bins-1)
    doses = self.get_dose_at_volume(roi, rel_vols)
    return np.append(doses, 0.)

def get_current_wrapper(self, what):
    return get_current(what)

def get_roi_mask(self, roi_name):
    plan = get_current('Plan')
    dgr = plan.TreatmentCourse.TotalDose.GetDoseGridRoi(RoiName = roi_name)
    mask = dgr.RoiVolumeDistribution.VoxelIndices
    return mask

def get_roi_relative_volumes(self, roi_name):
    plan = get_current('Plan')

    dgr = plan.TreatmentCourse.TotalDose.GetDoseGridRoi(RoiName = roi_name)
    mask = dgr.RoiVolumeDistribution.VoxelIndices


def get_roi_names(self):
    beam_set = get_current('BeamSet')
    sub_structure_set = beam_set.DependentSubStructureSet
    roi_names = [r.OfRoi.Name for r in sub_structure_set.RoiStructures if r.PrimaryShape is not None]
    roi_names = [n for n in roi_names if n not in ["External"]]
    return roi_names
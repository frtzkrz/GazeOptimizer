import numpy as np
from connect import *
from typing import List

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
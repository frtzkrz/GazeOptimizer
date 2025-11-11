import h5py
import numpy as np
from optimizer_functions.helpers import *
import math


def save_dose_distributions(self):
    gaze_angles = define_gaze_angle_grid(delta_polar=self.delta_polar, max_polar_deg=self.max_polar_deg)
    bin_size = 10 # [cGy]
    
    with h5py.File(self.h5py_file_path, "a") as f:
        gaze_group = f.require_group('gaze_angles')
        mask_group = f.require_group('roi_masks')
        roi_names = self.get_roi_names()

        #
        for gaze_angle in gaze_angles:

            #Calculate dose for current gaze angle
            self.set_gaze_angles(gaze_angle)
            self.calculate_dose()
            
            #Create Group '(polar, azimuthal)'
            angle_group = gaze_group.require_group(str(gaze_angle))

            #Save whole dose distribution and save to h5
            plan_dose = self.get_dose()
            angle_group.create_dataset("total_dose", data=plan_dose)


            
            #Get DVH for each ROI and save to h5
            for roi in roi_names:
                doses = self.get_dvh(roi)
                angle_group.create_dataset(roi, data=doses)
        



"""            sub_structure_set = beam_set.DependentSubStructureSet
            roi_names = [r.OfRoi.Name for r in sub_structure_set.RoiStructures if r.PrimaryShape is not None]
            roi_group = gaze_group.require_group("roi_masks")
            for roi in roi_names:
                dgr = plan.
                roi_group.create_dataset(roi, data=)"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py

from patient_functions.patient import Patient

def main():

    #patient id
    patient_id = '23129'

    #define folder to save plots
    plot_folder = f'plots/ray_dvhs_comparison'

    #initialize patient
    pat = Patient(patient_id=patient_id, h5py_file_path=f'results/{patient_id}/{patient_id}_9_angles.h5')
    
    pat.compare_dvhs(ray_path=f'results/{patient_id}/{patient_id}_ray_dvhs_0_0.txt', plot_folder=plot_folder)

if __name__ == "__main__":
    main()
    

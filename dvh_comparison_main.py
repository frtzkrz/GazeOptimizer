import argparse

import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py

from patient_functions.patient import Patient

def main():

    #patient id
    patient_id = 'P23336'

    #define folder to save plots
    plot_folder = f'plots/{patient_id}'

    #initialize patient
    pat = Patient(patient_id=patient_id, h5py_file_path=f'results/{patient_id}_5.h5')
    print(pat.optimal_angle_key)
    with h5py.File(pat.h5py_file_path, "r") as h5_file:
        total_dose = h5_file['gaze_angles'][pat.optimal_angle_key]['total_dose'][:]
    
    pat.compare_dvhs(ray_path=f'results/{patient_id}_ray_dvhs.txt', total_dose=total_dose)

    start_time = time.time()
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    

if __name__ == "__main__":
    main()
    

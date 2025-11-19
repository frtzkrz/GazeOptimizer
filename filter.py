import argparse

import pandas as pd
import matplotlib.pyplot as plt
import time

from patient_functions.patient import Patient

def main():

    #patient id
    patient_id = 'P23336'

    #define folder to save plots
    plot_folder = f'plots/{patient_id}'
    pat = Patient(patient_id=patient_id, h5py_file_path=f'results/{patient_id}/{patient_id}_9_angles.h5')
    filters = [
        {'roi_name': 'Macula', 'filter_type': 'D', 'value': 90, 'max_val': 40}, #D90_Macula < 40 Gy
        {'roi_name': 'Lens', 'filter_type': 'V', 'value': 10, 'max_val': 15} #V10_Lens < 15
        ]

    filtered_gaze_angles=pat.apply_dvh_filters(filters)
    

    start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    plt.show()

if __name__ == "__main__":
    main()

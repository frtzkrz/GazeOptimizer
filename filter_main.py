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

    #initialize patient
    pat = Patient(patient_id=patient_id, h5py_file_path=f'results/{patient_id}_5.h5')
    print(np.shape(pat.roi_mask_dict['Macula']))

    """#define filter criteria
    filter_dict = {
        'name': 'D90_Macula', 
        'roi': 'Macula', 
        'filter_type': 'D', 
        'value': 90, 
        'max': 40}  # Example filter criteria: D90 of Macula less than 40 Gy
    
    filter_dict_2 = {
        'name': 'V10_Lens', 
        'roi': 'Lens', 
        'filter_type': 'V', 
        'value': 10, 
        'max': 20}  #Create second filter: V10 of Lens less than 20%


    start_time = time.time()
    
    #first round of filtering
    filtered_keys = pat.full_filtered_metric_dvh_plot(filter_dict=filter_dict, plot_folder=plot_folder, save_fig=True)

    #second round of filtering based on first round
    filtered_keys_2 = pat.full_filtered_metric_dvh_plot(filter_dict=filter_dict_2, plot_folder=plot_folder, save_fig=True, filtered_gaze_angle_keys=filtered_keys)

    #find new optimal gaze angles after filtering
    new_opt = pat.find_new_optimal_gaze_angles(filtered_gaze_angle_keys=filtered_keys_2)['New Optimum']
    print(f"New optimal gaze angle after filtering: {new_opt} with cost {pat.costs_dict[new_opt]}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    plt.show()"""

if __name__ == "__main__":
    main()
    

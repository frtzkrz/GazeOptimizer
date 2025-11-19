import argparse

import pandas as pd
import matplotlib.pyplot as plt
import time

from optimizer_functions.optimizer import GazeOptimizer

def main():

    parser = argparse.ArgumentParser(description="Optimize angles using GazeOptimizer.")
    args = parser.parse_args()



    patient_id = '17213'
    delta_polar = 5
    max_polar_deg = 25
    weights = {'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1}

    Optimizer = GazeOptimizer(
        patient_id = patient_id,
        delta_polar=delta_polar,
        h5py_file_path=f'results/{patient_id}_{delta_polar}.h5',
        )

    
    start_time = time.time()


    #Optimizer.full_search()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    #Optimizer.full_scatter_plot()
    #Optimizer.full_metric_dvh_plot()


if __name__ == "__main__":
    main()
    

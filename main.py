import argparse

from functions.optimizer import GazeOptimizer
import pandas as pd
from functions.helpers import *
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(description="Optimize angles using GazeOptimizer.")
    args = parser.parse_args()



    patient_id = 23336
    delta_polar = 5
    max_polar_deg = 25
    weights = {'D2_Macula': 3, 'D20_OpticalDisc': 3, 'D20_Cornea': 1, 'V55_Retina':1, 'V27_CiliaryBody': 1, 'D5_Lens': 1}

    Optimizer = GazeOptimizer(
        patient_id = patient_id,
        extensive_path = f'save_eval/{patient_id}.csv',
        use_precalculated = True
        )
    #Optimizer.optimize()
    Optimizer.full_plot('scatter')
    plt.show()
    

if __name__ == "__main__":
    main()
    

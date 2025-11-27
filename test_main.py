from patient_functions.patient import *


def main():
    patient_id = 'P23336'
    angle_1 = '(25, 180)'
    angle_2 = '(25, 45)'
    pat = Patient(patient_id=patient_id, h5_file_path=f'results/{patient_id}/{patient_id}_9_angles.h5')

    plot_weight_search(patient=pat, gaze_angle_key_1=angle_1,gaze_angle_key_2=angle_2, n_steps=10)


if __name__ == "__main__":
    main()
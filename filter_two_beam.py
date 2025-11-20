from patient_functions.patient import *
import argparse
import json


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Perform search for two beam angles using cost function")
    parser.add_argument("--patient_id", type=str, help="Patient ID", default="23129")
    parser.add_argument("--h5py_path", type=str, help="Path to h5py file", default="results/23129/23129_delta_60.h5")
    parser.add_argument("--weight_path", type=str, help="Path to weight file", default="weights.json")
    n_steps = 10

    args = parser.parse_args()
    gaze_angle_dvhs_list = []
    pat = Patient(patient_id=args.patient_id, h5py_file_path=args.h5py_path)

    with h5py.File(pat.h5py_file_path, 'r') as h5_file:
        for i, angle_1 in enumerate(pat.gaze_angle_keys):
            dose_1 = h5_file[angle_1][:]
            for j, angle_2 in enumerate(pat.gaze_angle_keys[i+1:]):
                dose_2 = h5_file[angle_2][:]
                for w in np.linspace(0, 1, n_steps):
                    g = GazeAngleDVHs(
                        patient = pat,
                        angle_key=angle_1,
                        angle_key_2=angle_2,
                        dose=w*dose_1 + (1-w)*dose_2,
                        weight=w,
                        )
                    g.calculate_cost()
                    gaze_angle_dvhs_list.append(g)
    for roi in pat.roi_names:
        fig, ax = plt.subplots()
        for g in gaze_angle_dvhs_list:
            g.roi_dvhs[roi].plot(ax=ax)
        ax.set_title(f'ROI: {roi}')
        plt.show()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("")
    print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')

if __name__ == "__main__":
    main()

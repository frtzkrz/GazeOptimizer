import pandas as pd
import h5py

def fill_cache_from_csv(self):
    with open(self.extensive_path, mode="r", newline="", encoding="utf-8") as file:
        df = pd.read_csv(file, header = [0,1])
    self.cache = {(p, a): c for p, a, c in zip(df[('gaze_angles', 'polar')], df[('gaze_angles', 'azimuthal')], df[('total_cost', 'cost')])}

def get_row_by_gaze_angle(self, gaze_angle):
    polar, azimuthal = gaze_angle
    mask = (self.cost_df[('gaze_angles', 'polar')] == polar) & (self.cost_df[('gaze_angles', 'azimuthal')] == azimuthal)
    return self.cost_df[mask]

def fill_cache_from_h5py(self):
    self.cache={}
    try:
        with h5py.File(self.h5py_file_path, 'r') as f:
            gaze_angles = f['gaze_angles']
            for angle_key in gaze_angles.keys():
                gaze_angle = tuple(f['gaze_angles'][angle_key].attrs['gaze_angle'])
                self.cache[gaze_angle] = f['gaze_angles'][angle_key].attrs['total_cost']
    except FileNotFoundError:
        return


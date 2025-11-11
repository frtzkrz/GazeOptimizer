from typing import List, Tuple
from optimizer_functions.helpers import *

def grid_search(
    self, 
    ) -> Tuple[Tuple[float, float], float]:
    """
    Performs grid search for defined angles
    angles: list of angles (polar, azimuthal)

    Returns: angles and cost of best performing gaze angle
    """
    gaze_angles = self.define_gaze_angle_grid()
    l = len(gaze_angles)
    for i, gaze_angle in enumerate(gaze_angles):
        if gaze_angle not in self.cache:
            self.cost(gaze_angle)
        print_progress_bar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    best_angle, best_cost = min(self.cache.items(), key=lambda x: x[1])
    return best_angle, best_cost


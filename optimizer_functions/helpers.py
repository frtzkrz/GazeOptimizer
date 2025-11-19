#Contains helper functions that don't belong to the class GazeOptimizer
import numpy as np

def define_gaze_angle_grid(
        self
    ):

    """
    Creates list of dict of gaze angles spreading the plane
    delta_polar: determines the spacing between points
    """
    delta_polar = self.delta_polar
    max_polar_deg = self.max_polar_deg
    angles = [(0, 0)]
    polars = np.linspace(0, max_polar_deg, int(np.ceil(max_polar_deg/delta_polar)))
    for p in polars:
        if not p==0:
            azimuthals = np.round(np.linspace(0, 360, int(np.ceil(2*np.pi*p/delta_polar))),1)
            for a in azimuthals:
                angles.append((p, a))

    return angles



# Print iterations progress
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({iteration}/{total})', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def print_progress_opt(gaze_angle):
    print(f'\rTrying gaze_angle: {gaze_angle}')
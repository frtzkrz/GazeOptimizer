def calculate_gaze_angle(self, gaze_angle):
    """
    To be called if new gaze angle has to be examined. Calculates Cost, Metrics, DVHs, Dose
    """
    self.expensive_cost(gaze_angle=gaze_angle)



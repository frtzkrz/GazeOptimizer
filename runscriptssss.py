# Script recorded 26 Nov 2025, 08:44:36

#   RayStation version: 14.0.100.0
#   Selected patient: ...

from connect import *

beam_set = get_current("BeamSet")


with CompositeAction('Edit azimuthal gaze angle (GazeOptimizer, beam set: GazeOptimizer)'):

  beam_set.Beams['GazeOptimizer'].GazeAngles.AzimuthalGazeAngle = 15

  # CompositeAction ends 


# Script recorded 18 Nov 2025, 11:59:19

#   RayStation version: 14.0.100.0
#   Selected patient: ...

from connect import *

beam_set = get_current("BeamSet")


with CompositeAction('Edit azimuthal gaze angle (GazeOptimizer, beam set: GazeOptimizer)'):

  beam_set.Beams['GazeOptimizer'].GazeAngles.AzimuthalGazeAngle = 50

  # CompositeAction ends 


import sys
import os

# Print the paths to ensure we're in the right environment
print(sys.path)
print("Python Path:", sys.executable)
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    print("FilterPy is working!")
except ImportError as e:
    print("ImportError:", e)

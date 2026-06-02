import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Training.archs.vn_arch
import Training.models.vn_model
from basicsr.test import test_pipeline

if __name__ == '__main__':
    test_pipeline(root_path=os.path.dirname(os.path.abspath(__file__)))

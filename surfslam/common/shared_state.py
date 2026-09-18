"""
File: src/common/shared_state.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import torch.multiprocessing as mp

class SharedState:
    def __init__(self):
        self.last_mapped_frame_time = mp.Value('d', 0.)
        # Set to 1 when the posegraph backend reports IMU initialization complete
        self.backend_initialized = mp.Value('i', 0)

        # Frame processing statistics counters
        self.stereo_input_count = mp.Value('i', 0)
        self.stereo_processed_count = mp.Value('i', 0)
        self.registration_processed_count = mp.Value('i', 0)

import os
import time
import cv2
import numpy as np
import torch.multiprocessing as mp

from common.signals import Signal, StopSignal
from common.settings import Settings
from mapping.stereo.stereo_estimation_defom import DefomStereoEstimator
from common.frame import Frame
from common.timing import log_process_timing

class StereoMatcher:
    def __init__(self,
                 settings: Settings,
                 calibration: Settings,
                 stereo_process_signal: Signal,
                 completed_frame_signal: Signal,
                 mapping_input_signal: Signal):

        self._stereo_slot = stereo_process_signal.register()
        self._completed_frame_signal = completed_frame_signal
        self._mapping_input_signal = mapping_input_signal

        self._settings = settings
        self._shared_state = None  # Will be set in run()

        if self._settings.model_type == "DEFOM":
            self._stereo_estimator = DefomStereoEstimator(
                self._settings, calibration, "cuda:0"
            )
        else:
            raise RuntimeError("Unknown stereo estimator:", self._settings.model_type)

        self._processed_stop_signal = mp.Value("i", 0)
        self._term_signal = mp.Value("i", 0)
        self._frame_count = 0

    def start(self):
        pass

    def update(self):
        
        if self._stereo_slot.has_value():
            n = len(self._stereo_slot)
            for _ in range(n):
                frame: Frame | StopSignal = self._stereo_slot.get_value()

                if frame is None:  # Frame was dropped due to age
                    continue

                if isinstance(frame, StopSignal):
                    if not self._processed_stop_signal.value:
                        self._completed_frame_signal.emit(StopSignal())
                        self._mapping_input_signal.emit(StopSignal())
                    self._processed_stop_signal.value = 1
                    return

                stereo_pair = frame.stereo_image
                tic = time.perf_counter()
                depth_image = self._stereo_estimator.infer(stereo_pair)
                toc = time.perf_counter()

                timestamp = getattr(stereo_pair, "timestamp", None)
                if timestamp is not None:
                    try:
                        timestamp = float(timestamp)
                    except (TypeError, ValueError):
                        timestamp = str(timestamp)

                log_process_timing(
                    getattr(self._settings, "log_directory", None),
                    "stereo_inference",
                    toc - tic,
                    metadata={
                        "frame_id": getattr(frame, "_id", None),
                        "timestamp": timestamp,
                    },
                )


                if self._settings.debug.log_depth_images:
                    log_dir = self._settings.log_directory
                    log_path = f"{log_dir}/depth_images/frame_{self._frame_count}.png"
                    os.makedirs(f"{log_dir}/depth_images", exist_ok=True)

                    image = depth_image.image.detach().cpu().squeeze().numpy()
                    np.savez_compressed(log_path.replace('.png', '.npz'), depth=image)
                    image = np.where(image > 30.0, 0.0, image)

                    max_val = image.max()
                    if max_val > 0.0:
                        image = image / max_val

                    image_u8 = (image * 255.0).astype(np.uint8)
                    colorized = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)

                    cv2.imwrite(log_path, colorized)
                    
                
                if self._settings.debug.log_raw_images:
                    log_dir = self._settings.log_directory
                    log_path = f"{log_dir}/rgb_frames/frame_{self._frame_count}"
                    os.makedirs(log_path, exist_ok=True)

                    left = (frame.stereo_image.left_image.permute(1,2,0).cpu().numpy()).astype(np.uint8)
                    right = (frame.stereo_image.right_image.permute(1,2,0).cpu().numpy()).astype(np.uint8)

                    cv2.imwrite(f"{log_path}/left.png", left)
                    cv2.imwrite(f"{log_path}/right.png", right)

                frame.depth_image = depth_image
                self._completed_frame_signal.emit(frame.clone())
                self._mapping_input_signal.emit(frame.clone())

                # Increment statistics counter if available
                if self._shared_state is not None:
                    self._shared_state.stereo_processed_count.value += 1

                self._frame_count += 1
        
        
    def run(self, shared_state):
        self._shared_state = shared_state
        while not self._processed_stop_signal.value:
            self.update()
            
        print("Stereo matcher finished")
        while not self._term_signal.value:
            time.sleep(0.1)
        
        print("Stereo matcher stopped")
    
    def finish(self):
        pass

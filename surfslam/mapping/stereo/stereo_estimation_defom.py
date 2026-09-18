import os
import sys
import torch
import hashlib
import pickle
from pathlib import Path

code_dir = os.path.dirname(__file__)
raft_dir = os.path.join(code_dir, os.pardir, os.pardir,
                        os.pardir, "submodules", "DEFOM-Stereo")
sys.path.append(raft_dir)

from common.sensors import StereoImage, Image
from defom_core.utils.utils import InputPadder
from defom_core.defom_stereo import DEFOMStereo


class DefomStereoEstimator:
    def __init__(self, settings, calibration, device, enable_cache=False):
        self._settings = settings

        self.device = device
        model = DEFOMStereo(settings.model_config)
        state = torch.load(
            os.path.expanduser(settings.model_config.restore_ckpt), map_location="cpu"
        )
        if "model" in state:
            state = state['model']
        to_load = {}
        for k, v in state.items():
            if k.startswith("module."):
                to_load[k[len("module."):]] = v
            else:
                to_load[k] = v
        model.load_state_dict(to_load)
        model.to(device)
        model.eval()

        self.model = model
        self.calibration = calibration
        self.bf = calibration['stereo']['baseline_m'] * calibration['stereo']['fx']

        self.ray_range = settings.ray_range
        # Hard cutoff on estimated depth; ray_range is scene-specific while this
        # bounds obviously-degenerate disparities.
        self.max_depth_m = settings.get('max_depth_m', 100.0)

        self._padder = None

        # Initialize cache directory (persistent across runs)
        cache_base = getattr(settings, 'cache_dir', None)
        if cache_base is None:
            # Default to user's home directory cache
            cache_base = Path.home() / ".cache" / "defom_stereo"
        else:
            cache_base = Path(cache_base)
        self.cache_dir = cache_base
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

    def _compute_cache_key(self, l_img, r_img, iters, scale_iters):
        """Compute a hash key for the given inputs."""
        # Create a hash from the tensor data and parameters
        hasher = hashlib.sha256()
        hasher.update(l_img.cpu().numpy().tobytes())
        hasher.update(r_img.cpu().numpy().tobytes())
        hasher.update(str(iters).encode())
        hasher.update(str(scale_iters).encode())
        return hasher.hexdigest()

    def _get_cached_disp(self, cache_key):
        """Try to load cached disparity from disk."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")
                return None
        return None

    def _save_cached_disp(self, cache_key, disp_pr):
        """Save disparity to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(disp_pr.cpu(), f)
        except Exception as e:
            print(f"Warning: Failed to save cache file {cache_file}: {e}")

    @torch.no_grad()
    def infer(self, stereo_image: StereoImage):
        torch.cuda.synchronize()

        stereo_image = stereo_image.clone().to(self.device)

        if self._padder is None:
            self._padder = InputPadder(
                stereo_image.left_image.shape, divis_by=32)
        l_img, r_img = self._padder.pad(stereo_image.left_image.unsqueeze(0),
                                        stereo_image.right_image.unsqueeze(0))

        # Check cache if enabled
        disp_pr = None
        if self.enable_cache:
            cache_key = self._compute_cache_key(
                l_img, r_img,
                self._settings.model_config.valid_iters,
                self._settings.model_config.scale_iters
            )
            disp_pr = self._get_cached_disp(cache_key)
            if disp_pr is not None:
                disp_pr = disp_pr.to(self.device)

        # Run model if not cached
        if disp_pr is None:
            disp_pr = self.model(
                l_img,
                r_img,
                iters=self._settings.model_config.valid_iters,
                scale_iters=self._settings.model_config.scale_iters,
                test_mode=True
            )

            # Save to cache if enabled
            if self.enable_cache:
                self._save_cached_disp(cache_key, disp_pr)

        disp = self._padder.unpad(disp_pr).squeeze(0).clamp(min=0)
        depth = self.bf / disp

        depth[disp == 0] = 0.
        depth[depth < self.ray_range[0]] = 0
        depth[depth > self.max_depth_m] = 0
        torch.cuda.synchronize()

        return Image(depth, stereo_image.timestamp)

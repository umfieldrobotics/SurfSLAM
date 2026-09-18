from typing import Dict, List
import numpy as np
import torch
import torch.multiprocessing as mp
import os

from common.frame import Frame
from common.pose import Pose, RegistrationResult
from common.shared_state import SharedState
from common.signals import Signal, StopSignal
from common.settings import Settings
from mapping.descriptors.matcher import SuperGlueMatcher
from mapping.place_recognition.image_descriptor import ImageDescriptorManager
from mapping.pointcloud_registration.feature_alignment_registration import (
    FeatureAlignmentRegistration,
)
from mapping.pointcloud_registration.gicp import run_gicp
from mapping.pointcloud_registration.pointcloud_cleaner import GPUPointCloudCleaner


class Registration:
    def __init__(
        self,
        settings: Settings,
        calibration: Settings,
        completed_frame_signal: Signal,
        registration_result_signal: Signal,
    ):

        self._settings = settings
        self._calibration = calibration
        self._completed_frame_slot = completed_frame_signal.register()
        self._registration_result_signal = registration_result_signal

        # Used to indicate to an external process that I've processed the stop signal
        self._processed_stop_signal = mp.Value("i", 0)
        # Set to 0 from an external thread when it's time to actuall exit.
        self._term_signal = mp.Value("i", 0)

        self._frames_by_id: Dict[int, Frame] = {}
        self._frames: List[Frame] = []

        self._matcher = None
        self._registration = None
        self._img_descriptor_manager = None
        self._shared_state = None  # Will be set in run()

    def start(self):
        if self._matcher is not None:
            return

        self._img_descriptor_manager = ImageDescriptorManager.from_file(
            self._settings.place_recognition.vocab_path  # type: ignore
        )

        self._settings["matching"]["log_directory"] = self._settings.log_directory
        self._matcher = SuperGlueMatcher(
            self._settings.matching,  # type: ignore
            self._calibration,
            self._img_descriptor_manager,
            device="cuda:0",
        )  # type: ignore

        if self._settings.cleanup.enabled:
            self._pointcloud_cleaner = GPUPointCloudCleaner(
                self._settings.cleanup, device="cuda:0"
            )

        if self._settings.registration_type.lower() == "ransac_3d":
            self._registration = FeatureAlignmentRegistration(
                self._settings.ransac_3d, self._calibration
            )
        elif self._settings.registration_type.lower() == "tmap":
            raise NotImplementedError("TMAP registration not yet implemented")
        else:
            raise ValueError("Unknown registration type")
        
    def _fast_voxel_downsample(self, points: torch.Tensor, voxel_size: float):
        if voxel_size is None:
            return points
        
        voxel_size = torch.tensor(voxel_size, device=points.device, dtype=points.dtype)
        min_pt = points.min(dim=0).values # center of bottom corner 
        
        voxel_min_bound = min_pt - 0.5 * voxel_size # need the corner of voxel
        coords = torch.floor((points - voxel_min_bound) / voxel_size).long() # what coordinate does each point fall into
        locations, inv_location_map = torch.unique(coords, dim=0, return_inverse=True)
        counts = torch.bincount(inv_location_map, minlength=locations.shape[0]).to(points.dtype)

        sums = torch.zeros((locations.shape[0], 3), device=points.device, dtype=points.dtype)
        sums.index_add_(0, inv_location_map, points)
        
        mean_pts = sums / counts[:, None]
        return mean_pts


    def refine_transform(self, frame, matched_frame, estimated_transform):

        # refine it with gicp
        # TODO (Seth) this is taking 2ms still, which still feels long
        target_pcd_full = self._fast_voxel_downsample(self._registration._frame_to_pcd(frame), self._settings.icp_refinement.downsample_resolution)
        source_pcd_full = self._fast_voxel_downsample(self._registration._frame_to_pcd(matched_frame), self._settings.icp_refinement.downsample_resolution)
        
        if self._settings.cleanup.enabled is False:
            return run_gicp(
                source_pcd_full.cpu().numpy(),
                target_pcd_full.cpu().numpy(),
                estimated_transform,
                self._settings.icp_refinement.max_correspondence_distance,
                self._settings.icp_refinement.max_iterations,
                None,
                self._settings.icp_refinement.num_threads,
            )
        cleaned_source_pcd, _ = self._pointcloud_cleaner.clean(source_pcd_full)
        cleaned_target_pcd, _ = self._pointcloud_cleaner.clean(target_pcd_full)

        gicp_result = run_gicp(
            cleaned_source_pcd.cpu().numpy(),
            cleaned_target_pcd.cpu().numpy(),
            estimated_transform,
            self._settings.icp_refinement.max_correspondence_distance,
            self._settings.icp_refinement.max_iterations,
            None,
            self._settings.icp_refinement.num_threads,
        )

        return gicp_result

    def update(self):
        if self._matcher is None:
            self.start()
        if self._completed_frame_slot.has_value():
            n = len(self._completed_frame_slot)
            for _ in range(n):

                frame: Frame | StopSignal = self._completed_frame_slot.get_value()

                if frame is None:  # Frame was dropped due to age
                    continue

                if isinstance(frame, StopSignal):
                    if not self._processed_stop_signal.value:
                        self._registration_result_signal.emit(StopSignal())

                    self._processed_stop_signal.value = 1
                    return

                # Increment counter for frames that reach registration module
                if self._shared_state is not None:
                    self._shared_state.registration_processed_count.value += 1

                if len(self._frames) > 0 and not self._settings.frame_to_frame.enabled:
                    # in the case where there are frames but we DONT want to do frame-to-frame tracking,
                    # we still need to notify the matcher of it for caching
                    self._matcher.register_frame(frame)
                elif len(self._frames) > 0:
                    prev_time = self._frames[-1].get_time()
                    new_frame_time = frame.get_time()

                    if self._settings.debug.draw_frame_matches:
                        log_dir = f"{self._settings.log_directory}/frame_matches/"
                        os.makedirs(log_dir, exist_ok=True)
                        debug_path = f"{log_dir}/{frame.get_id()}_matched_{self._frames[-1].get_id()}.png"
                    else:
                        debug_path = None

                    # compute the correspondences between the last frame and the new frame
                    mkpts0, mkpts1, mconf, *_ = self._matcher.match(
                        self._frames[-1], frame, debug_path=debug_path
                    )

                    # SuperGlue returns numpy arrays; convert them to tensors for TEASER++
                    mkpts0 = torch.as_tensor(mkpts0, dtype=torch.float32)
                    mkpts1 = torch.as_tensor(mkpts1, dtype=torch.float32)

                    if self._settings.debug.frame_to_frame_registration:
                        debug_dir = f"{self._settings.log_directory}/frame_to_frame_registration"
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = f"{debug_dir}/frame_{frame.get_id()}_matched_{self._frames[-1].get_id()}.pkl"
                    else:
                        debug_path = None

                    estimated_transform, inlier_mask = self._registration.solve(
                        self._frames[-1], mkpts0, frame, mkpts1, debug_file=debug_path
                    )

                    passfail = (
                        inlier_mask.sum() >= self._settings.frame_to_frame.min_inliers
                        and inlier_mask.mean()
                        >= self._settings.frame_to_frame.min_inlier_ratio
                    )

                    if self._settings.debug.draw_teaser_inliers:
                        log_dir = (
                            f"{self._settings.log_directory}/teaser/frame_to_frame/"
                        )
                        passfail_str = "PASS" if passfail else "FAIL"
                        log_dir = f"{log_dir}/{passfail_str}"
                        log_path = f"{log_dir}/{frame.get_id()}_matched_{self._frames[-1].get_id()}.png"
                        os.makedirs(log_dir, exist_ok=True)

                        data = {
                            "mkpts0": mkpts0.numpy(),
                            "mkpts1": mkpts1.numpy(),
                            "inlier_mask": inlier_mask,
                            "frame_id": frame.get_id(),
                            "matched_frame_id": self._frames[-1].get_id(),
                        }
                        with open(log_path.replace(".png", ".pkl"), "wb") as f:
                            import pickle

                            pickle.dump(data, f)

                        self._matcher.visualize_matches(
                            self._frames[-1],
                            frame,
                            mkpts0,
                            mkpts1,
                            mconf,
                            passfail=passfail,
                            inliers=inlier_mask,
                            out_path=log_path,
                        )

                    if passfail:
                        gicp_res = self.refine_transform(
                            self._frames[-1], frame, estimated_transform
                        )

                        estimated_transform = torch.from_numpy(
                            gicp_res["transformation"]
                        ).float()
                        cov = torch.from_numpy(gicp_res["covariance"]).float()

                        gicp_passfail = (
                            gicp_res["fitness"]
                            > self._settings.icp_refinement.min_gicp_fitness
                            and gicp_res["inlier_rmse"]
                            < self._settings.icp_refinement.max_inlier_rmse
                        )

                        if not gicp_passfail:
                            print(
                                f"Rejected GICP refinement. Fitness: {gicp_res['fitness']:.03f}, Inlier RMSE: {gicp_res['inlier_rmse']:.03f}"
                            )
                            continue

                        result = RegistrationResult(
                            prev_time,
                            new_frame_time,
                            Pose(estimated_transform.clone(), cov=cov),
                            id_a=self._frames[-1]._id,
                            id_b=frame._id,
                        )

                        self._registration_result_signal.emit(result)

                    else:
                        print(
                            f"Rejected TEASER registration. Num Inliers: {inlier_mask.sum()}. Inlier Ratio: {inlier_mask.mean():.03f}"
                        )

                self._frames.append(frame)
                self._frames_by_id[frame.get_id()] = frame
                if self._settings.matching.loop_closure.enabled:
                    all_loop_candidates = self._matcher.get_loop_candidates(
                        frame, debug=self._settings.debug.draw_loop_candidates,
                        n_candidates=self._settings.matching.loop_closure.num_candidates
                    )

                    if all_loop_candidates is not None:
                        all_inlier_ratios = [c[-1].mean() for c in all_loop_candidates]
                        search_order = np.argsort(-np.array(all_inlier_ratios))
                        for candidate_idx in search_order:
                            best_candidate = all_loop_candidates[candidate_idx]
                            matched_id, mkpts0, mkpts1, mconf, _ = best_candidate
                            matched_frame = self._frames_by_id[matched_id]
                            if self._settings.debug.loop_closure_registration:
                                debug_dir = f"{self._settings.log_directory}/loop_closure_registration"
                                os.makedirs(debug_dir, exist_ok=True)
                                debug_path = f"{debug_dir}/loop_frame_{frame.get_id()}_matched_{matched_id}.pkl"
                            else:
                                debug_path = None
                                
                            estimated_transform, inlier_mask = self._registration.solve(
                                matched_frame,
                                torch.from_numpy(mkpts1),
                                frame,
                                torch.from_numpy(mkpts0),
                                debug_file=debug_path,
                                threshold=self._settings.matching.loop_closure.ransac_threshold
                            )

                            passfail = inlier_mask.sum().item() >= self._settings.matching.loop_closure.teaser_min_inliers and \
                                       inlier_mask.mean().item() >= self._settings.matching.loop_closure.teaser_min_inlier_ratio

                            if self._settings.debug.draw_teaser_inliers:
                                passfail_str = "PASS" if passfail else "FAIL"
                                log_dir = f"{self._settings.log_directory}/teaser/loop_closures/{passfail_str}/"
                                log_path = f"{log_dir}/{frame.get_id()}_matched_{matched_frame.get_id()}.png"
                                os.makedirs(log_dir, exist_ok=True)

                                self._matcher.visualize_matches(
                                    frame,
                                    matched_frame,
                                    mkpts0,
                                    mkpts1,
                                    mconf,
                                    passfail,
                                    inliers=inlier_mask,
                                    out_path=log_path,
                                )

                            if passfail:
                                gicp_res = self.refine_transform(
                                    matched_frame, frame, estimated_transform
                                )

                                refined_estimated_transform = torch.from_numpy(gicp_res["transformation"]).float()
                                cov = torch.from_numpy(gicp_res["covariance"]).float()

                                loop_closure = RegistrationResult(
                                    matched_frame.get_time(),
                                    frame.get_time(),
                                    Pose(refined_estimated_transform, cov=cov),
                                    matched_frame.get_id(),
                                    frame.get_id(),
                                )
                                print(f"Emitting loop closure from frame {matched_frame.get_id()} to {frame.get_id()}")
                                self._registration_result_signal.emit(loop_closure)
                                break

    def run(self, shared_state: SharedState) -> None:
        self.start()

        self._shared_state = shared_state
        self._last_mapped_frame_time = shared_state.last_mapped_frame_time

        while not self._processed_stop_signal.value:
            self.update()

        print("Registration done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        # This is used to prevent race conditions at shutdown
        while not self._term_signal.value:
            continue
        print("Exiting registration process.")

    def finish(self):
        pass

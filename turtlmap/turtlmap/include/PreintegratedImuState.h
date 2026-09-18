#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>
#include <memory>

namespace turtlmap {

/**
 * @brief Saved state for gtsam::PreintegratedCombinedMeasurements
 *
 * Uses deep copy to preserve the complete state of the PreintegratedCombinedMeasurements object.
 * This is simpler and more reliable than serialization.
 */
struct PreintegratedImuState
{
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> pim_copy;

  PreintegratedImuState() = default;
};

/**
 * @brief Save the state of a PreintegratedCombinedMeasurements object
 *
 * Creates a deep copy of the PreintegratedCombinedMeasurements object.
 *
 * @param pim The PreintegratedCombinedMeasurements to save
 * @return State struct containing the deep copy
 */
inline PreintegratedImuState savePreintegratedImuState(const gtsam::PreintegratedCombinedMeasurements& pim)
{
  PreintegratedImuState state;
  state.pim_copy = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(pim);
  return state;
}

/**
 * @brief Load state into a PreintegratedCombinedMeasurements object
 *
 * Returns a copy of the saved PreintegratedCombinedMeasurements object.
 *
 * @param state The saved state to restore
 * @return New PreintegratedCombinedMeasurements with the restored state
 */
inline gtsam::PreintegratedCombinedMeasurements loadPreintegratedImuState(const PreintegratedImuState& state)
{
  if (!state.pim_copy) {
    return gtsam::PreintegratedCombinedMeasurements();
  }
  return *state.pim_copy;
}

/**
 * @brief Alternative: Load state by updating an existing object via pointer
 *
 * This version updates an existing PreintegratedCombinedMeasurements object
 * by copying from the saved state.
 *
 * @param pim Pointer to the PreintegratedCombinedMeasurements to update
 * @param state The saved state to restore
 */
inline void loadPreintegratedImuState(gtsam::PreintegratedCombinedMeasurements* pim, const PreintegratedImuState& state)
{
  if (pim == nullptr || !state.pim_copy) {
    return;
  }
  *pim = *state.pim_copy;
}

}  // namespace turtlmap

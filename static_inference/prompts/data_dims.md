# Model output dims and their semantics

## Pi05 Base
Path: `checkpoints/openpi/base/pi05_base_torch`
(download again to confirm)

Valid: 8 dims (0-7)
- 0-6: single-arm Franka; deltas 
- 7: gripper (binary)

The checkpoint may contain franka_robocasa_padded norm stats which are 12 dims instead, but all dims after dim 7 are pads that you can directly remove and disregard.

# Robocasa Dataset dims and their semantics
Path: `datasets/robocasa365/atomic-seen-splits`

Valid: 12 dims (0-11)
- 0-2 `base_motion`: base velocities vx, vy, yaw
- 3 `base_motion`: torso lift delta (constant 0 everywhere)
- 4 `control_mode`: controller switch
- 5-7 `end_effector_position`: arm EEF position delta (dx, dy, dz)
- 8-10 `end_effector_rotation`: arm EEF orientation delta
- 11 `gripper_close`: : gripper (binary)
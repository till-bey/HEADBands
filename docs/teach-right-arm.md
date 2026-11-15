# Panther Teach & Playback Node (Dual Arm + Gripper + Base Lift)

Node implementation: [teach.py](../src/panther_examples/panther_examples/teach.py)

---

## Complete Pipeline & Usage

### 1. Check Status
```bash
ros2 run panther_examples teach
```
Displays active/inactive controllers, CAN hardware components (`unix_openarm_left`, `unix_openarm_right`, `raw_can0`, `raw_can1`), and current lift state.

---

### 2. Disable Arms & Interactively Adjust Base Height
```bash
ros2 run panther_examples teach adjust-height
```
* Drops torque on both arms and grippers so they can be repositioned freely.
* Launches an interactive console prompt to jog or set torso height:
  * `w` / `+`: raise base (+0.05 m)
  * `s` / `-`: lower base (-0.05 m)
  * `<number>`: set exact height in meters (e.g. `0.35` between `0.0` and `0.8`)
  * `q` / `enter`: finishes adjustment, re-enables arm drives, and activates holding control at the new height.

---

### 3. Disable Motors for Direct Manual Teaching
```bash
ros2 run panther_examples teach disable --yes
```
Drops holding torque on all 16 arm joints and both grippers so they can be guided by hand.

---

### 4. Record Dual-Arm Motion
```bash
ros2 run panther_examples teach record --seconds 10
```
Records:
* Left arm (`joint_la0` .. `joint_la7`)
* Right arm (`joint_ra0` .. `joint_ra7`)
* Left gripper (`joint_lf`) & Right gripper (`joint_rf`)
* Live torso lift height (`body_joint`)
* Saved into `teach_out/recording.json`.

---

### 5. Generate Action Sequence with Optional Base Height
```bash
# Option A: Convert with a specific target base height
ros2 run panther_examples teach action --height 0.45

# Option B: Convert using the recorded height
ros2 run panther_examples teach action
```
Downsamples trajectory to 10 Hz and packages:
* Target `initial_lift_height`
* Left and right arm waypoints with synchronized gripper positions
* Saved to `teach_out/action.json`.

---

### 6. Synchronous Playback
```bash
# Play default action.json (or action_right.json fallback)
ros2 run panther_examples teach play --yes

# Or play a specific file (e.g. frisbee throw)
ros2 run panther_examples teach play --yes --file teach_out/action_right.json
```
1. **Base Lift:** If an initial lift height is set, moves torso to height and pauses for completion.
2. **Drives:** Re-enables left and right arm CAN drives and gripper hardware.
3. **Dual Trajectory:** Concurrently dispatches left and right arm trajectories to their respective `FollowJointTrajectory` action servers.
4. **Gripper Streaming:** Streams synchronized gripper commands to both `/left_gripper/gripper_command` and `/right_gripper/gripper_command`.
5. **Reset:** Switches back to holding controller `dual_arm_forward_position_controller` once finished.

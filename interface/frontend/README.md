# Swiss Multilingual Voice & Robotics Web UI

Zero-credential browser client for the Swiss Multilingual Voice & Robotics Assistant.

## Features
- **Real-time 16 kHz 16-bit Mono PCM Capture:** Uses Web Audio API downsampling over `ws(s)://<host>/ws/voice`.
- **Voice Activity Detection & Barge-In Interruption:** Visual RMS meter, partial transcripts, sentence-chunked WAV playback, and instant playback/generation cancellation.
- **Multilingual & Dialect Controls:** Supports `auto`, `gsw` (all 8 SwissDial dialects including `vs` Wallisertiitsch), `de-DE`, `de-CH`, `fr-FR`, and `fr-CH`.
- **Gemini Robotics ER 2 & Safety Console:** Camera image upload with 2D spatial detection overlay, mandatory human confirmation cards, and immediate Emergency Stop (`E-STOP`).
- **Live Robot Camera** (`assets/live_camera.js`): **Start live camera** repeatedly calls `POST /api/v1/robotics/camera/capture`, one request at a time (at most about 3 frames/s), and shows `/camera/color/image_raw` from the ROS 2 bridge. Starting it may auto-launch the Orbbec driver on the robot, as the **Trigger Robot Camera** button does; the bridge stops the driver after its idle timeout. It pauses while the tab is hidden.
- **Simulation View** (`assets/robot_view.js`): draws the Panther from `assets/robot_description.urdf` with the joint positions from `GET /api/v1/robotics/joints`, which the bridge streams at up to 5 Hz from `/joint_states` plus the lift and gripper states. Run the bridge against the HEADBands mock sim (`ros2 launch panther_sim mock_sim.launch.py`) to watch the sim: on the same machine, from a `docker/run.sh --sim` shell, since the sim is loopback-only on `ROS_DOMAIN_ID=77`, or against the robot (15) to watch the robot. Drag to rotate, scroll to zoom; **Demo** animates the joints with no bridge. `robot_description.urdf` is an unmodified copy of HEADBands `src/panther_sim/urdf/robot_description.urdf` (branch `clean_up`); the gesture planner uses the same file in `backend/app/robotics/data/`. Copy both again if that model changes.

- **Generated Gestures** (Gesture box, voice, or ER 2): describe any arm gesture ("salute", "wave hello", "point at the door", "raise both arms"). Gemini returns hand keyframes only; `backend/app/robotics/gesture_planner.py` solves IK on the URDF, refuses unreachable poses, caps joint speed at 0.45 rad/s and keeps 3 cm from approximate head, neck and torso boxes. The plan previews in Simulation and waits for confirmation like every other motion; the bridge re-checks limits and speed and plays it through `/master_controller/joint_trajectory`.

The camera and simulation views only read. Commands reach the robot only after a human confirms them.

## Security
Contains **zero** cloud API keys or service-account credentials. Served directly by FastAPI (`SERVE_FRONTEND=true`) or any static web server.

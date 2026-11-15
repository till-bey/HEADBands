# Swiss Multilingual Voice & Robotics Web UI

Zero-credential browser client for the Swiss Multilingual Voice & Robotics Assistant.

## Features
- **Real-time 16 kHz 16-bit Mono PCM Capture:** Uses Web Audio API downsampling over `ws(s)://<host>/ws/voice`.
- **Voice Activity Detection & Barge-In Interruption:** Visual RMS meter, partial transcripts, sentence-chunked WAV playback, and instant playback/generation cancellation.
- **Multilingual & Dialect Controls:** Supports `auto`, `gsw` (all 8 SwissDial dialects including `vs` Wallisertiitsch), `de-DE`, `de-CH`, `fr-FR`, and `fr-CH`.
- **Gemini Robotics ER 2 & Safety Console:** Camera image upload with 2D spatial detection overlay, mandatory human confirmation cards, and immediate Emergency Stop (`E-STOP`).
- **Live Robot Camera** (`assets/live_camera.js`): **Start live camera** repeatedly calls `POST /api/v1/robotics/camera/capture`, one request at a time (at most about 3 frames/s), and shows `/camera/color/image_raw` from the ROS 2 bridge. Starting it may auto-launch the Orbbec driver on the robot, as the **Trigger Robot Camera** button does; the bridge stops the driver after its idle timeout. It pauses while the tab is hidden.
- **Simulation View** (`assets/robot_view.js`): draws the Panther from `assets/panther.urdf` with the joint positions from `GET /api/v1/robotics/joints`, which the bridge streams at up to 5 Hz from `/joint_states` plus the lift and gripper states. Run the bridge against the HEADBands mock sim (`ros2 launch panther_sim mock_sim.launch.py`) to watch the sim: on the same machine, from a `docker/run.sh --sim` shell, since the sim is loopback-only on `ROS_DOMAIN_ID=77`, or against the robot (15) to watch the robot. Drag to rotate, scroll to zoom; **Demo** animates the joints with no bridge. `panther.urdf` is a kinematics-only copy of HEADBands `src/panther_sim/urdf/robot_description.urdf`; regenerate it if that model changes.

Both views only read. They send no robot commands.

## Security
Contains **zero** cloud API keys or service-account credentials. Served directly by FastAPI (`SERVE_FRONTEND=true`) or any static web server.

# Operating the UniX AI Panther

How to charge, power, connect, and teleoperate the Panther at the hackathon. Procedures and specs below come from the Panther Robot User Guide (UniX AI, 28 Jan 2026) unless a sentence says otherwise. The English manual is a rough translation; where a label is garbled, this page says so.

Software commands, topics, and SSH live in [the ROS 2 README](../ros2/README.md).

## Shift-start checklist

1. Indoor floor, flat, dry, away from stairs and ledges. Clear a path wider than the robot (chassis about 510 mm, arms spanning about 650 mm).
2. Nothing in front of the arms. On shutdown the arms lose power and drop.
3. Charger is the one supplied with the robot. Charge fully before the first job of the day.
4. Someone knows where the physical emergency stop is, and someone is holding the wireless emergency stop, before the base moves.
5. Router is the lab network in [Network](#network) (SSID `UNIX-TEST2`).
6. Power on and wait for the voice prompts (about 40 seconds after the lights come on). Then confirm it answers a voice query, which is the manual’s “is it online” check.

## Safety rules that matter on the floor

- Indoor only. The manual says rain, soil, and dust can get into the machine.
- Stay off stairs, ramps without railings, and slopes steeper than 6°. Step-over clearance is 20 mm.
- Use the supplied power cord only. Keep the cord and the charge port dry. Do not use a cord, plug, or socket that is damaged, corroded, or hot.
- Do not open or modify the battery (48 V, 18 Ah, lithium iron phosphate).
- Lift the robot by the abdomen or under the light strip at the bottom of the body. Do not lift it by the head. The manual warns the head can detach.
- Do not twist or push the head by hand.
- Do not push the robot while it is driving.
- Do not grab a moving arm. Clear the volume in front of the arms before power-off.
- People stay out of the arm reach (about 982 mm per arm) while anyone is on the VR controllers or publishing commands.
- Working temperature in the spec table is −10 °C to 40 °C.

## What arrived in the kit

| Item | Qty |
| --- | --- |
| Robot | 1 |
| Charger | 1 |
| Battery (cell) | 1 |
| Certificate | 1 |
| Software license agreement | 1 |

VR teleop expects a Meta Quest 3s, listed separately in the manual. It is not in the configuration-list table.

## Size, power, and motion limits

These are the numbers that change how you stage the robot. The manual’s example column is the high-end configuration.

| Item | Value |
| --- | --- |
| Overall size | 650 mm front-to-back × 650 mm across the arms × 1600 mm tall |
| Chassis width | 510 mm each side, as printed |
| Track (left-right wheelbase) | 320 mm |
| Wheelbase | 420 mm |
| Drive | 4-wheel drive, four-wheel steering, permanent-magnet direct drive |
| Wheel motors | 160 W × 4, rated torque 8 N·m × 4 |
| Top speed | 2 m/s |
| Turning | 0 mm in spin mode |
| Slope / step | 6° / 20 mm |
| Runtime | 6–12 h (the table says “duration of flight”) |
| Charge time | about 2.5 h |
| Battery | 48 V / 18 Ah LiFePO4, direct charge, electronic parking brake |
| Payload row | The table has a 100 kg figure whose English label is corrupted (“Conduct oneself with dignity”). Treat 100 kg as an unlabeled spec until someone checks the Chinese source. Do not assume it is payload or mass. |
| Torso travel | 800 mm, motion space 632–1432 mm |
| Computer | NVIDIA Jetson AGX Orin 64 GB (275 TOPS) |
| Arms | 8 degrees of freedom each, about 7 kg each, 982 mm reach, 6 kg max end load, 12 kg both arms together, repeatability ≤ 0.5 mm, two-finger gripper |
| Head | yaw ±90°, pitch 20° up and 86° down |
| Sensing | Orbbec Gemini 2 RGB-D, binocular HD module, LiDAR × 2, ultrasonic × 8, six-mic ring array |
| E-stop | physical and wireless |
| Base computer I/O called out on the chassis | HDMI × 1, USB × 1, RJ45 × 1, debug × 1, 24 V power out × 1 |

The Orin itself exposes more ports (USB 3.0 × 6, CAN × 2, Ethernet × 2, HDMI × 1). The chassis plate you can reach from behind is the smaller set in the photo.

## Rear panel

Stand behind the robot. The controls are on the right side, above the chassis.

The photo in Part 2 of the user guide shows a rectangular plate:

- Top left: HDMI
- Top right: network (RJ45). A small pinhole sits under that jack; the manual does not say what it does.
- Bottom left: USB
- Bottom right: 24 V output, for sensors
- To the right of the plate: round power button with the power symbol. The Chinese label beside it is 开关 (“switch”).

The charging port is a separate round connector on the side of the chassis, next to a wheel. In the user-guide photo it is the port circled in red, with a small round indicator beside it. Blue light strips run along the body.

## Charging

The spec table calls the method direct charge. The manual also says to place the charging dock flat against a wall.

1. Put the supplied charger against a wall on a flat floor.
2. Clear 0.5 m on both sides and 1.5 m in front of it.
3. Plug the charger into mains, then into the round charge port on the side of the chassis.
4. Fully charge before first use. Budget about 2.5 hours from empty.
5. Unplug by the plug, with the cord free of water and out of the drive path.

If the robot will sit unused for a long time: charge it, power it off, and only then remove the battery, with the power switch already off. The manual says battery removal is done under professional guidance. Store cool and dry, out of sun and humidity.

## Power on

1. Clear the floor in front of and beside the arms.
2. Press and hold the round switch on the rear-right of the chassis for 3–5 seconds.
3. Blue indicators on the head, chest, and waist come on.
4. Wait about 40 seconds. The robot should say that the voice AI assistant has been activated, then that the self-check completed.

Startup is finished only after that second prompt. The ROS stack comes up with the boot; see the ROS README for how to tell that the controllers are active.

## Power off

1. Move anything the arms could hit. The arms drop when power is removed.
2. Press and hold the same switch for 3–5 seconds.
3. The indicator goes out. That is power-off.

A frozen VR headset is a different button: hold the Quest power button for 10 seconds. That restarts the headset, not the robot.

## Network

The robot is operated over the local LAN. The manual’s prescribed router setup is:

| Setting | Value |
| --- | --- |
| SSID | `UNIX-TEST2` |
| Security | WPA2-PSK |
| Wi-Fi password | `Passw0rd123` |
| Router LAN address | `192.168.15.254` |
| Mask | `255.255.255.0` |
| DHCP pool | `192.168.15.2` – `192.168.15.253` |

The English text says to set the “gateway” to `255.255.255.0`. That number is a subnet mask. Use it as the mask. The guide does not give a separate gateway address; on this LAN the router is `192.168.15.254`.

Recommended radio, if you have a choice: tri-band Wi-Fi 6E (2.4 + 5 + 6 GHz), about 3000 Mbps, 4×4 or 8×8 MIMO. The robot’s own radio is listed as Wi-Fi 6 (802.11 ax/ac/a/b/g/n) on 2.4 GHz, minimum 50 Mbps, plus Bluetooth 5.1. The Quest 3s is Wi-Fi 6E.

The guide does not print the robot’s IP. After the router is up, read the client list and look for the industrial PC. Addresses in `192.168.15.2`–`192.168.15.253` are the pool.

### See that the robot joined Wi-Fi

Either of these, from the manual:

- The robot’s computer shows up in the router’s client list.
- The voice assistant answers a weather question. The printed wake phrase is “Wanda Wanda”, which looks like a bad translation. Try a weather question once the boot prompts have finished; if it answers, the manual counts that as online.

SSH, once you have the IP (password is the vendor default from the SDK, `unixai`):

```bash
ssh unix_ai@192.168.15.XX
```

## VR teleop (Meta Quest 3s)

App name in the manual: **Data Acquisition Version**.

Open the headset library with the controller (the manual’s figure 1) and launch that app. The manual assumes the app is already installed. It does not give a store URL.

Three panels:

1. **Host management.** Connect to the robot server. This is the data link. If the address field is empty, enter the robot IP. “Invalid IP address format” means type it again.
2. **Head.** Turn Head on to drive head rotation from the headset.
3. **Remote operation.** Puts the robot into a preset pose, then tracks your arms.

### Controller map

| Control | Effect |
| --- | --- |
| Right B | Turn teleop on or off |
| Controller pose | Arm pose |
| Trigger | Gripper |
| Either stick, up/down | Base forward / back |
| Either stick, left/right | Base sideways |
| Right stick, left/right | Base yaw |
| Right stick, up/down | Torso height |
| Left grip (side button) | Speed step |
| Right grip (side button) | Video-mode step |

The stick descriptions overlap (both sticks strafe, and the right stick also yaws and lifts). If two functions fight, treat the on-screen labels in the headset as newer than this table and move slowly.

### Entering teleop

Poses:

- **Standby.** Robot waits.
- **Repositioning.** Arms return to the initial pose.
- **Vertical.** Both hands face inward. This is the pose you match before tracking.

Steps, as printed:

1. Choose vertical or free pose. The status line should track the robot.
2. In the second menu, choose both arms, left, or right.
3. Raise your hands to 90° from your body so you match the robot.
4. Wait until the arm in the video is fully up and the prompt is green **Standby**.
5. Press right B to hide the panel and start tracking. After B is pressed, the instruction says the arm cannot be moved — read that as: tracking owns the arm, so do not fight it.
6. Press right B again to leave teleop. The robot returns to standby, and you can move freely.
7. To resume, choose Free Pose and repeat.

Status phrases:

| Prompt | Meaning |
| --- | --- |
| In posture preparation | Moving into the preset pose. Wait. |
| Await the opportune moment | Waiting for the next command. |
| Reduction position | Back at the reset pose. |
| Vertical orientation | In the vertical pose. Match it with your body. |

### Headset shortcuts

| Action | Input |
| --- | --- |
| Reset view | Hold the right controller’s key for 2 seconds (the manual’s table leaves the key name blank) |
| Wake / sleep | Short-press headset power |
| Headset power | Hold headset power 2 seconds |
| Force-restart headset | Hold headset power 10 seconds |
| IPD | Roller on the headset |
| Photo | Hold the right controller key and press the trigger |
| Dead UI | Press headset power twice |

Quest 3s specs from the manual: Android 12 / Quest OS, 90 Hz (120 Hz supported), 1832 × 1920 per eye.

### Teleop faults

| Symptom | What the manual says to do |
| --- | --- |
| Frozen UI, buttons dead | Hold headset power 10 seconds |
| Bad IP format | Re-enter the robot IP |
| Wi-Fi off or dropped | Headset Settings, and the router |
| Empty address | Enter the robot IP |
| Robot got no data | Headset is not receiving the robot. Check the robot-side remote module |
| Left or right arm “collided with an object” | Something is in that arm’s range. Clear it and move carefully |
| Controller battery low | Battery door is at the arrow on the handle. Press and slide down, swap the cell |
| No image in the headset | Press headset power twice |

## Driving limits to brief anyone on the sticks

- 2 m/s is the documented maximum. Start far below that.
- The base can spin in place.
- Electronic brake holds it when parked.
- Eight ultrasonics and two LiDARs are the obstacle sensors. A service can turn the ultrasonics off; leave them on unless the ROS README’s service is being used on purpose.
- Arm collision warnings from the headset mean stop and look, then move the obstacle.

## What this manual does not specify

- The robot’s DHCP address, hostname, and any LED code for charge state.
- Which blue light means charging versus powered.
- Where the physical emergency-stop button and the wireless fob are on this unit. Find both before the base moves.
- A battery-percent topic or a low-battery behavior.
- How the “charging dock” and the side charge port fit together beyond “place the charger, plug into the circled port.”

#!/usr/bin/env python3
"""Read-only: the CiA402 state, mode and error code of each arm drive, over each joint's sdo_read service.

  ros2 run panther_examples can_state              # right arm
  ros2 run panther_examples can_state --arm both

Changes nothing on the robot. Needs canopen_interfaces (in the Docker image). The sim has no drives.
"""
import argparse

import rclpy

from panther_examples.robot import LEFT_ARM, RIGHT_ARM, can_service

STATUSWORD, MODE_DISPLAY, ERROR_CODE = 0x6041, 0x6061, 0x603F
# CiA402 statusword: (mask, value, state)
STATES = [(0x4F, 0x00, "not ready to switch on"), (0x4F, 0x40, "switch on disabled (no torque)"),
          (0x6F, 0x21, "ready to switch on (no torque)"), (0x6F, 0x23, "switched on (no torque)"),
          (0x6F, 0x27, "OPERATION ENABLED (torque)"), (0x6F, 0x07, "quick stop active"),
          (0x4F, 0x0F, "fault reaction active"), (0x4F, 0x08, "FAULT")]
MODES = {1: "profile position", 3: "profile velocity", 4: "profile torque", 6: "homing", 7: "interpolated position",
         8: "cyclic sync position", 9: "cyclic sync velocity", 10: "cyclic sync torque"}


def drive_state(statusword):
    return next((name for mask, value, name in STATES if statusword & mask == value), f"unknown {statusword:#06x}")


def mode_name(display):
    m = display & 0xFF
    m = m - 256 if m > 127 else m  # int8
    return MODES.get(m, f"mode {m}")


def main():
    from canopen_interfaces.srv import CORead  # here, so the decoding above is testable without it
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right", "both"), default="right")
    joints = {"left": LEFT_ARM, "right": RIGHT_ARM, "both": LEFT_ARM + RIGHT_ARM}[ap.parse_args().arm]
    rclpy.init()
    node = rclpy.create_node("can_state")

    def read(joint, index):
        cli = node.create_client(CORead, can_service(joint, "sdo_read"))
        if not cli.wait_for_service(timeout_sec=5.0):
            return None
        f = cli.call_async(CORead.Request(index=index, subindex=0))
        rclpy.spin_until_future_complete(node, f, timeout_sec=5.0)
        node.destroy_client(cli)
        return f.result().data if f.done() and f.result().success else None

    for j in joints:
        sw, mode, err = (read(j, i) for i in (STATUSWORD, MODE_DISPLAY, ERROR_CODE))
        print(f"{j}:  {drive_state(sw) if sw is not None else 'no reply':32s} | "
              f"{mode_name(mode) if mode is not None else 'mode ?':22s} | "
              f"error {f'{err:#06x}' if err is not None else '?'}" + (f" | statusword {sw:#06x}" if sw is not None else ""))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

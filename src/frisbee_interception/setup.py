from glob import glob

from setuptools import find_packages, setup

package_name = "frisbee_interception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HEADBands",
    description="Frisbee detection, tracking and interception prediction (prediction-only by default)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "phone_camera = frisbee_interception.phone_camera_node:main",
            "frisbee_detector = frisbee_interception.detector_node:main",
            "frisbee_tracker = frisbee_interception.tracker_node:main",
            "frisbee_interceptor = frisbee_interception.interceptor_node:main",
            "frisbee_catch_trigger = frisbee_interception.catch_trigger_node:main",
            "frisbee_catch_pose = frisbee_interception.catch_pose_cli:main",
            "frisbee_offline = frisbee_interception.offline:main",
        ],
    },
)

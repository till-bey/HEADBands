from setuptools import find_packages, setup

package_name = "swiss_robot_bridge"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "websockets>=13.1",
        "pydantic>=2.9.0",
        "Pillow>=10.4.0",
        "numpy>=1.26.0",
    ],
    zip_safe=True,
    maintainer="Leukerbad Robotics Team",
    maintainer_email="robotics@example.ch",
    description="UniX AI Panther local ROS 2 safety gateway, camera trigger, and outbound Cloud Run bridge",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge_node = swiss_robot_bridge.bridge_node:main",
        ],
    },
)

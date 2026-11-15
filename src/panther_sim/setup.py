from glob import glob

from setuptools import setup

package_name = "panther_sim"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/urdf", glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HEADBands",
    maintainer_email="headbands@localhost",
    description="Panther arms, head, lift and grippers on ros2_control mock hardware, with the robot's names.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot_topics = panther_sim.robot_topics:main",
        ],
    },
)

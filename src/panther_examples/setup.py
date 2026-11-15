from setuptools import setup

package_name = "panther_examples"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HEADBands",
    maintainer_email="headbands@example.com",
    description="Short command publishers for the Panther.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "commands = panther_examples.commands:main",
            "teach = panther_examples.teach:main",
            "cartesian_follow = panther_examples.cartesian_follow:main",
            "joint_check = panther_examples.joint_check:main",
            "master_gui = panther_examples.master_gui:main",
            "can_state = panther_examples.can_state:main",
            "pickup = panther_examples.pickup:main",
        ],
    },
)

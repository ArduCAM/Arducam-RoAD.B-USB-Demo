from glob import glob
import os

from setuptools import setup


package_name = "arducam_uvc_stereo_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            os.path.join("share", "ament_index", "resource_index", "packages"),
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="arducam",
    maintainer_email="arducam@arducam.com",
    description="ROS 2 Demo for Arducam UVC stereo cameras.",
    entry_points={
        "console_scripts": [
            "stereo_camera_node = arducam_uvc_stereo_ros.driver_node:main",
            "list_devices = arducam_uvc_stereo_ros.list_devices:main",
        ],
    },
)

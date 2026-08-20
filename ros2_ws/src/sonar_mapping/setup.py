from setuptools import find_packages, setup
import os
from glob import glob

package_name = "sonar_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adolfo Damiano Cafaro",
    maintainer_email="adaca@dtu.dk",
    description="AKAZE leading-edge seabed mapping from forward-looking sonar + odometry.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"sonar_mapping_node = {package_name}.sonar_mapping_node:main",
        ],
    },
)

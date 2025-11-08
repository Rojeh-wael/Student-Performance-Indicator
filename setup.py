from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

setup(
    name="student-performance-indicator",
    version="0.0.1",
    author="Rojeh Wael",
    author_email="rojehwael@yahoo.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
  
)

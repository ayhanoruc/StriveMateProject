from setuptools import find_packages, setup
import sys, os
from typing import List


def get_requirements() -> List[str]:
    file_path = "requirements.txt"
    requirement_list = []
    with open(file_path) as f:
        for line in f:
            req = line.strip()
            if not req.startswith("#") and (req != "") and (req != "-e ."):
                requirement_list.append(req)

    return requirement_list


setup(
    name="ChefApp",
    version="0.0.1",
    description="ChefApp is the future of kitchen-copilot apps",
    author="ayhanoruc",
    author_email="ayhan.orc.2554@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
)
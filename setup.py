from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    with open(file_path) as f:
        requirements = f.read().splitlines()
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='ensemble_project', 
    version='0.0.1',
    author = "Shivanjali Kaveti",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
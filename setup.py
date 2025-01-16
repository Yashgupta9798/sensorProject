from setuptools import find_packages, setup  #setup tools is a package that provide tools for packaging python projects and find_packages is a function which helps in finding the packages in our projects
from typing import List

HYPEN_E_DOT = '-e.'

def get_requirements(file_path : str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)  #since -e. is not package it is only used when we install package without using setup and directly from requirements.txt
    return requirements


setup(
    name = 'Fault detection',
    version = '0.0.1',
    author = 'yash',
    author_mail = 'yashgupta9934@gmail.com',
    install_requirements = get_requirements('requirements.txt'),
    packages = find_packages()
)
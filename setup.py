from setuptools import setup, find_packages
from typing import List #In this List we can specify the datatypes of the elements stored


# Declaring variables for setup function
PROJECT_NAME = "banking-prediction"
VERSION = "0.0.1"
AUTHOR = "Utkarsh Tewari"
DESCRIPTION ="This project is for bank credit risk prediction"
LICENSE = "Apache License Version 2.0"
REQUIREMENT_FILE_NAME = "requirements.txt"


# Declaring a function to fetch packages from requirements.txt
def get_requirements_list() -> List[str]:
    """This function is responsible for fetching all the packages
    specified in requirements.txt file.
    
    Returns
    -------
    required_packages : List[str]
            List of the names of the packages.
    """
    with open(REQUIREMENT_FILE_NAME, 'r') as requirement_file:
            required_packages = []
            for i in requirement_file.readlines():
                if i[:-1]!='' and i[:-1]!='-e.':
                    required_packages.append(i[:-1])
            return required_packages


# find_packages(): returns all the packages(user_defined) with __init__.py 
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    license=LICENSE,
    packages=find_packages(),
    install_requires=get_requirements_list()
)




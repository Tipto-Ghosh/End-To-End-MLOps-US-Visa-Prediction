from setuptools import setup , find_namespace_packages
from typing import List


# read the README.md file
with open("README.md" , "r" , encoding = "utf-8") as file:
    long_description = file.read()


HYPEN_E_DOT = '-e .'

# read the requirements.txt file
def get_requirements(requirements_file_path: str) -> List[str]:
    requirements = []
    
    with open(requirements_file_path) as req:
        requirements = req.readlines()
        # remove the newline sysmbol
        requirements = [req.replace('\n' , '') for req in requirements]
        
        # Also remove the -e .
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements 


# All meta-data's
project_name = "End-To-End-MLOps-US-Visa-Prediction"
version = "0.0.0"
author_name = "Tipto-Ghosh"
author_email = "tiptoghosh@gmail.com"
url = "https://github.com/Tipto-Ghosh/End-To-End-MLOps-US-Visa-Prediction"


setup(
    name = project_name,
    version = version,
    author = author_name,
    author_email = author_email,
    description = "End To End MLOps Machine Learing Project to predcit US visa accepted or not",
    long_description = long_description,
    long_description_content_type = "text/markdown", 
    url = url,
    packages = find_namespace_packages(),
    install_requires = get_requirements('requirements.txt') 
)
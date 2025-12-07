'''
setup.py is the configuration file used for packaging and distributing Python projects. It defines the project’s metadata, like name and version, and specifies which packages to include and which dependencies need to be installed.
In my projects, I often use setuptools.setup() with find_packages() to automatically detect packages, and I load dependencies from requirements.txt using a helper function. This way, when someone installs my project with pip install ., all the necessary dependencies are installed automatically."

"-e . in requirements.txt is used for editable installs, so developers can test changes without reinstalling the package each time
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
  
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

print(get_requirements())

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Vaishnavi Nagane",
    author_email="vaishnavinagane986@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
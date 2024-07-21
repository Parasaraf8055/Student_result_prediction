from setuptools import find_packages,setup
from typing import List


h_e_d = '-e.'
global get_requiremnets
def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return list of requirements
    
    '''
    requirement = []
    with open(file_path) as obj:
        requirements = obj.readlines()
        requirements  = [req.replace('/n',' ') for req in requirements]
        if h_e_d in requirements:
            requirements.remove(h_e_d)

    return requirements        

setup(

    name = 'mlproject',
    version='0.0.1',
    author='paras',
    author_email='sarafparas792@gmail.com',
    packages=find_packages(),

    ##insted of doing this we wil crete a function and give the path

    ##install_requires = ['pandas','numpy','seaborn']

    install_requires = get_requirements('requirements.txt')

)
#Este archivo es el responsable en crear mi modelo de ML como un paquete
#Tambien sirve para instalarlo en otros proyectos y se pueda usar.

from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOR= "-e ."

def get_requirements(file_path:str) ->List[str]:
    '''
    Esta funcion retornara una lista que contiene los paquetes a instalar
    '''
    requirements=[]
    with open('requirements.txt') as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOR in requirements :
            requirements.remove(HYPEN_E_DOR)
    return requirements





setup(
name= 'mlproject',
version='0.0.1',
author='Mateo Heras',
author_email='wmateohv@hotmail.com',
packages= find_packages(),
install_requires=get_requirements('requirements.txt')

)
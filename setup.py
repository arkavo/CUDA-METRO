import os
from distutils.core import setup
from setuptools import find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
	# Name of the package 
	name='CUDA-METRO',
	version="1.2.1",
	packages=find_packages(),
	# Short description of your library 
	description='pyCUDA Metropolis Monte Carlo 2D Heisenberg Model Simulation',
	# Long description of your library 
	long_description=long_description,
	long_description_content_type='text/markdown',
	# Your name 
	author='Arkavo Hait',
	# Your email 
	author_email='arkavohait@gmail.com',
	# Either the link to your github or to your website 
	url='https://github.com/arkavo/CUDA-METRO',
	# Link from which the project can be downloaded 
	download_url='https://github.com/arkavo/CUDA-METRO/releases/',
	py_modules=["montecarlo"],
	include_package_data=True,
	package_data={'': ['montecarlo.py']},
	# List of keywords 
	keywords=['python',
			   'cuda',
			   'pycuda',
			   'gpu',
			   'parallel computing',
			   'computational physics',
			   ''],
	# List of packages to install with this one 
	install_requires=[	'pycuda==2024.1.2',
						'numpy==2.1.2',
						'seaborn>=0.13.2',
						'tqdm>=4.66.5'],
	# https://pypi.org/classifiers/ 
	classifiers=['Development Status :: 4 - Beta',
	'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1' ]
)

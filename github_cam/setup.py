from setuptools import setup, find_packages
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'cam',
     version = '0.0.1',
     description = 'Python package for Libera WFOV Camera',
     long_description = long_description,
     classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Satellite Remote Sensing',
        ],
     keywords = 'Libera WFOV Camera',
     url = 'https://github.com/hong-chen/cam',
     author = 'Hong Chen, K. Sebastian Schmidt',
     author_email = 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
     license = 'GPLv3',
     packages = find_packages(),
     install_requires = [
         'numpy',
         'scipy',
         'h5py',
         'matplotlib',
         'pysolar',
         ],
     python_requires = '~=3.9',
     # scripts = ['bin/sks2h5', 'bin/sks2txt'],
     include_package_data = True,
     zip_safe = False
     )

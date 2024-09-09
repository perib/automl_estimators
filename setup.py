#!/usr/bin/env python
# -*- coding: utf-8 -*-
#TODO update this
from setuptools import setup, find_packages


def calculate_version():
    initpy = open('automl_estimators/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version


package_version = calculate_version()

setup(
    name='automl_estimators',
    python_requires='<3.12', #for configspace compatibility
    version=package_version,
    author='Pedro Ribeiro',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/tpot2',
    license='MIT License', #TODO
    # entry_points={'console_scripts': ['tpot2=tpot2:main', ]},
    description=('Estimator Wrappers for Model Selection/Hyperparameter Optimization Tools'),
    long_description='''
Wraps Optuna and AMLTK into an sklearn estimator for ease of use.
''',
    zip_safe=True,
    install_requires=['numpy==1.26.4',
                      'scipy>=1.3.1',
                      'scikit-learn>=1.3.0',
                      'update_checker>=0.16',
                      'joblib>=1.1.1',
                      'xgboost>=1.7.0',
                      'optuna>=3.0.5',
                      'configspace>=0.7.1',
                     ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning'],
)

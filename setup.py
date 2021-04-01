from distutils.core import setup

INSTALL_REQUIRES = open("requirements.txt").readlines()

setup(
    name='lime_timeseries',
    version='0.1',
    install_requires = INSTALL_REQUIRES,
    description='Applying the LIME algorithm for explainable machine learning to time series data',
    url='https://github.com/emanuel-metzenthin/Lime-For-Time',
    py_modules=[
        'lime_timeseries'
    ],
)

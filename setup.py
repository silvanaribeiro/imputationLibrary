from setuptools import find_packages, setup

setup(
    name='imputationLibrary',
    packages=find_packages(),
    include_package_data=True,
    version='0.0.46',
    description='Imputation Library for time series Data.',
    author='Silvana Mara Ribeiro',
    license='GNU 3',
    install_requires=[
        'matplotlib==3.1.0',
        'numpy==1.19.1',
        'pandas==0.24.2',
        'scikit-learn==0.23.1',
        'scipy==1.5.2',
        'seaborn==0.9.0',
        'xgboost==1.1.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
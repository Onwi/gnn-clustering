from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pooling_genomic',
    packages=find_packages('src'),
    install_requires=required,
    package_dir={'': 'src'},
    version='0.0.1',
    description='Pooling GNNs for Genomic Data Classification and Interpretation',
)
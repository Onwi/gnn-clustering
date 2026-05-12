from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='gnn_clustering',
    packages=find_packages('src'),
    install_requires=required,
    package_dir={'': 'src'},
    version='0.1.0',
    description='GNN-based clustering and classification models',
    author='Your Name',
    license='MIT',
)

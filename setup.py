from setuptools import setup, find_packages


setup(
    name='ezCV',
    version='0.0',
    packages=find_packages(),
    author='Frederico Caroli',
    description='Backend library for ezCV',
    python_requires='>=3',
    install_requires=['numpy', 'pyyaml', 'opencv-python>=3']
)

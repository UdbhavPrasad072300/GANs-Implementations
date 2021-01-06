from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3 :: Only',
]

setup(
    name='gans_implementations',
    version='0.0.1',
    description='A bunch of GAN implementations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Udbhav Prasad',
    author_email='udbhavprasad072300@gmail.com',
    url='https://github.com/UdbhavPrasad072300/GANs-Implementations',
    license='MIT',
    classifiers=classifiers,
    package=find_packages(),
    package_dir={'': 'src'},
)

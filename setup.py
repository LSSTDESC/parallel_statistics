from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='parallel_statistics',
    version='0.10',
    description='Calculating basic statistics in parallel, incrementally',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LSSTDESC/parallel_statistics',  # Optional
    author='Joe Zuntz',  # Optional
    author_email='joe.zuntz@ed.ac.uk',  # Optional
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='MPI, statistics',  # Optional
    packages=find_packages(where='.'),  # Required
    python_requires='>=3.6',
    install_requires=['numpy'],
)
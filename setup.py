from setuptools import setup, find_packages

setup(
    name="MeanFieldGame",
    version="0.1",
    packages=find_packages(),
    description="A package for mean field game analysis with graphons",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'pandas',
        'matplotlib',
        'seaborn'
    ],
)
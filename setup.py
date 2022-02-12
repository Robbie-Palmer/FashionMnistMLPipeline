from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    requirements = requirements_file.read()
setup(
    name='fashmnistmlpipeline',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Robbie-Palmer/FashionMnistMLPipeline',
    install_requires=requirements
)

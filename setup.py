from setuptools import setup, find_packages


def get_requirements(file_path: str) -> list[str]:

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

            return requirements
setup(
    name="Chennai Transit AI",
    version="0.1.0",
    packages=find_packages(),
    author="Muralidharan RMS",
    author_email= 'rmsmuralidharan@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="MIT"

)
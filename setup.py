from setuptools import setup, find_packages

setup(
    name="pit_advisor_libraries",
    version="1.0",
    packages=find_packages(where="./f1-pitstop-advisor/pit_advisor_libraries"),
    package_dir={"": "./f1-pitstop-advisor/pit_advisor_libraries"},
)
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="srl-local",
    version="0.0.1",
    author="openpsi-projects",
    author_email="openpsi.projects@gmail.com",
    description="open-source SRL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        "gym",
        "torch", 
        "wandb"
    ],
    entry_points={
        'console_scripts': [
            'srl-local = rlsrl.apps.main:main',
        ]
    }
)
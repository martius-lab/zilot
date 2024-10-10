from setuptools import find_packages, setup

setup(
    name="zilot",
    version="0.1.0",
    author="Thomas Rupf",
    author_email="thrupf@ethz.ch",
    description="Code for Zero Shot offline Imitation Learning via Optimal Transport",
    url="https://github.com/martius-lab/zilot",
    license="Apache",
    license_files=("LICENSE",),
    keywords=["Reinforcement Learning", "Imitation Learning", "Optimal Transport"],
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={"zilot": []},
    install_requires=[],
)

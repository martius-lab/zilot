from setuptools import find_packages, setup

setup(
    name="zilot",
    version="0.1.0",
    author="Thomas Rupf",
    author_email="tm.rupf@gmail.com",
    description="TODO",
    url="https://github.com/martius-lab/zilot",
    license="Apache",
    license_files=("LICENSE",),
    long_description="TODO",
    long_description_content_type="text/markdown",
    keywords=["RL", "IL"],
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={"zilot": []},
    install_requires=[],
)

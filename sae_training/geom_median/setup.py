import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geom_median",
    version="0.1.0",
    author="Krishna Pillutla",
    author_email="pillutla@cs.washington.edu",
    description="Implementation of the smoothed Weiszfeld algorithm to compute the geometric median",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krishnap25/geom_median",
    project_urls={
        "Bug Tracker": "https://github.com/krishnap25/geom_median/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.18.1',
        ]
)

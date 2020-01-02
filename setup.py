from setuptools import setup, find_packages

import glview


def read_deps(filename):
    with open(filename) as f:
        deps = f.read().split('\n')
        deps.remove("")
    return deps


def read_readme(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = lines[:-2]  # drop last two lines (CI build status)
        text = ''.join(lines)
        return text


setup(name="glview",
      version=glview.__version__,
      description="Lightning-fast image viewer with smooth zooming & panning.",
      url="http://github.com/toaarnio/glview",
      author="Tomi Aarnio",
      author_email="tomi.p.aarnio@gmail.com",
      license="MIT",
      long_description=read_readme("README.md"),
      long_description_content_type="text/markdown",
      packages=find_packages(),
      include_package_data=True,
      install_requires=read_deps("requirements.txt"),
      python_requires=">=3.6",
      entry_points={"console_scripts": ["glview = glview:main"]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      zip_safe=True)

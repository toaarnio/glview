from setuptools import setup, find_packages

import glview


def read_deps(filename):
    with open(filename) as f:
        deps = f.read().split('\n')
        deps.remove("")
    return deps


setup(name='glview',
      version=glview.__version__,
      description='Lighting-fast image viewer with smooth zooming & panning.',
      url='http://github.com/toaarnio/glview',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=read_deps("requirements.txt"),
      entry_points={'console_scripts': ['glview = glview:main']},
      zip_safe=True)

from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('pyfoil/__init__.py').read(),
)[0]

setup(name='pyfoil',
      version=__version__,


      description="This repo contains basic tools for working with airfoils, using pyspline as the backend.",
      author='',
      author_email='',
      url='https://github.com/mdolab/pyfoil',
      packages=[
          'pyfoil',
      ],
      install_requires=[
            'numpy>=1.16',
            'scipy>=1.2'
            'pygeo>=1.2.0',
            'pyspline>=1.1.0'
      ],
      classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python"]
      )

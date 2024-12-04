from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'library for training basic networks and analyzing landscape of cost function'
LONG_DESCRIPTION = 'README.md'

setup(name="forgetmenot", 
      version = VERSION,
      author = "amyzeth",
      author_email = "",
      description = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages = find_packages(exclude=('tests')),
      package_data={"":["*.json"]},
      install_requires = ["numpy >= 1.24.1", 
                          "torch >= 2.1.0"], 
      keywords = ['python'],
      classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux"
        ]
)
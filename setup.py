import sys
import setuptools
from setuptools.command.bdist_egg import bdist_egg

sys.path.append('src')
from version import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg
    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """
    def run(self):
        sys.exit("ERROR: aborting implicit building of eggs. Use \"pip install .\" to install from source.")

cmdclass = {'bdist_egg': bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled}

setuptools.setup(
    name             = "mrashpen",
    version          = __version__,
    author           = "Saikat Banerjee",
    author_email     = "bnrj.saikat@gmail.com",
    description      = "Provides methods for penalized regression of Mr.ASH",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license          = "MIT",
    url              = "https://github.com/banskt/mr-ash-pen",
    project_urls     = {
        "Bug Tracker": "https://github.com/banskt/mr-ash-pen/issues",
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir      = {"": "src"},
    packages         = ["mrashpen"],
    python_requires  = ">=3.9",
    install_requires = [
        "numpy>=1.19.4",
    ],
    cmdclass         = cmdclass,
)

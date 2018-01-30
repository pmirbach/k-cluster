from distutils.core import setup

setup(
    name = "edvar",
    packages = [
        "edvaraux",
    ],
    package_data={"edvaraux":["configspec"]},
    requires = ["scipy (>=0.11)",
                "numpy",
                "pickle",
                "pyqs"],
    version = "0.1",
    url = "",
    author = "Malte Schueler",
    author_email = "mschueler@itp.uni-bremen.de",
    description = "",
    long_description = """                                                                                
    """,
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        #"License :: ???",                                                                                
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ])

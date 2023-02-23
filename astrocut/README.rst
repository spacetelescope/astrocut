Introduction
------------
This document will serve to explain the core functionality of the 3 tools designed for creating TESS cubes and cutouts:
CubeFactory, TicaCubeFactory, and CutoutFactory.

`CubeFactory` (`make_cubes.py`)
------------
The `CubeFactory` class contains a library of methods used to generate a cube from TESS mission-delivered Science Processing 
Operations Center (SPOC) FFIs. 

TicaCubeFactory
------------
The `CubeFactory` class contains a library of methods used to generate a cube from TESS Image CAlibrator (TICA) quick-look FFIs.
Much of the logic mirrors that of `CubeFactory`, however, certain components are different due to the difference in 
file structure between the TICA FFIs and the SPOC FFIs.

CutoutFactory
------------
The `CutoutFactory` class contains a library of methods used to generate cutout files from a pre-generated TESS cube. This class
is agnostic to TICA and SPOC. 

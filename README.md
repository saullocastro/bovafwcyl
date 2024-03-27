Bayesian Optimization for VAFW CYLinders (bovafwcyl)
===

This respository contains the revised code originally created in Aravind Ashok's MSc thesis ( http://resolver.tudelft.nl/uuid:f0b2cdec-d5d8-4d98-83fe-da1f3ffc8d6d )
entitled "Bayesian Optimization for Lightweight Design of Variable Stiffness Composite Cylinder". We focused on variable-angle filament wound cylinders (VAFW).

The code has been cleaned-up and heavily simplified and re-structured.

The finite element being used to discretize the problem is the SC-BFSC, available on GitHub ( https://github.com/saullocastro/bfsccylinder ).

Before running this code
---
Make sure to install the following libraries:

    python -m pip install numpy scipy skopt sklearn composites bfsccylinder

How to run this code?
---
The main file to be run is the `bovafwcyl.py`:

    python bovafwcyl.py


Changing parameters
---
Geometry, inside `bovafwcyl.py`, look for the `geo_dict` variable:

        geo_dict = dict(
            L=0.300, # length
            R=0.15 # radius
        )


Material properties, inside `bovafwcyl.py`, look for the `mat_dict` variable:

        mat_dict = dict(
            E11=90e9,
            E22=7e9,
            nu12=0.32,
            G12=4.4e9,
            G23=1.8e9,
            plyt=0.4e-3 # ply thickness
        )

Number of layers and design loads, look for the `layers_loads` variable:

    layers_loads = (
                    (1,  50e3),
                    (2, 100e3),
                    (2, 200e3),
                    (2, 500e3),
                    (3, 200e3),
                    (3, 500e3),
                    (3, 1000e3),
                    (4, 500e3),
                    (4, 1000e3)
                    )

Discretization level, currently controlled by the number of elements along the
circumferential direction `ny`, look for the following variables, where the
defaul values are recommended, based on previous convergence analyses:

    ny_init_sampling = 55
    ny_optimization = 55
    ny_verification = 65


Authors
---

Aravind Ashok (original MSc thesis http://resolver.tudelft.nl/uuid:f0b2cdec-d5d8-4d98-83fe-da1f3ffc8d6d )

Saullo G. P. Castro (supervision, code cleanup, code curator)

Gad Marconi (code cleanup)

J. H. Almeida Jr. (validation, formal analysis)

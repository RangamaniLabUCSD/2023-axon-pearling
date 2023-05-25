#!/usr/bin/env python

import argparse

import polyscope as ps
import pymem3dg.visual as dg_vis
from pathlib import Path

import driver


parser = argparse.ArgumentParser(description="Visualize.")
parser.add_argument("filename", action="store", help="File to visualize")

args = parser.parse_args()

file = Path(args.filename)

if file.suffix == ".nc":
    dg_vis.animate(
        str(file),
        # parameters=driver.p,
        showBasics=True,
        showForce=False,
        showPotential=False,
        geodesicDistance=False,
        notableVertex=False,
        meanCurvature=True,
        edgeLength=True,
        vertexDualArea=True,
        gaussianCurvature=True,
        mechanicalForce=False,
        spontaneousCurvatureForce=False,
        deviatoricCurvatureForce=False,
        externalForce=False,
        capillaryForce=False,
        lineCapillaryForce=False,
        osmoticForce=False,
        adsorptionForce=False,
        aggregationForce=False,
        entropyForce=False,
        springForce=False,
        chemicalPotential=False,
        spontaneousCurvaturePotential=False,
        aggregationPotential=False,
        dirichletPotential=False,
        adsorptionPotential=False,
        entropyPotential=False,
        deviatoricCurvaturePotential=False,
    )
elif file.suffix == ".ply":
    dg_vis.visualizePly(
        str(file),
        "spontaneousCurvaturePotential",
        "lineCapillaryForce",
        "externalForce",
    )
else:
    raise RuntimeError("Unknown file type")
ps.show()

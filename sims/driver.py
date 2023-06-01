import pymem3dg as dg
import pymem3dg.util as dgu

from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing import Tuple
from functools import partial

import warnings

from units import unit

from tqdm.contrib.concurrent import process_map

import contextlib

import sys
import pdb


from parameter_variation import (
    osmolarities,
    tensions,
    bending_moduli,
    target_volume_scale,
    reservoir_volume,
)

warnings.filterwarnings("error")


def constantSurfaceTensionModel(area: float, tension: float) -> Tuple[float, float]:
    """Constant surface tension model

    Args:
        area (float): total surface area of the mesh. Unused
        tension (float): value of surface tension

    Returns:
        Tuple[float, float]: surface tension and associated energy
    """
    energy = tension * area
    return (tension, energy)


def ambientSolutionOsmoticPressureModelWUnit(
    volume: float,
    RT: unit.Quantity,
    enclosed_solute: unit.Quantity,
    ambient_concentration: unit.Quantity,
    reservoir_volume: unit.Quantity = 0 * unit.micron**3,
) -> Tuple[float, float]:
    """Compute the pressure

    Args:
        volume (float): Volume in micrometer**3
        RT (float): Value of RT in (nanonewton * micrometer / Kelvin / attomole)
        enclosed_solute (float): attomole amount
        ambient_concentration (float): attomol/micrometer**3

    Returns:
        Tuple[float, float]: pressure and energy
    """
    volume = volume * unit.micron**3
    # print(
    #     f"Scaled Kv strength (iRTn): {(RT*enclosed_solute /reservoir_volume.magnitude).to(unit.micron*unit.nanonewton)}"
    # )
    RT = RT / reservoir_volume.magnitude
    pressure = RT * (enclosed_solute / volume - ambient_concentration)  # kPa

    ratio = ambient_concentration * volume / enclosed_solute
    energy = RT * enclosed_solute * (ratio - np.log(ratio) - 1)
    # print(
    #     "-" * 20,
    #     volume,
    #     pressure.to(unit.nanonewton / unit.micron**2),
    #     energy.to(unit.nanonewton * unit.micron),
    #     (enclosed_solute / volume).to(unit.molar),
    #     sep="\n",
    # )
    return (pressure.magnitude, energy.magnitude)


def getGeometryParameters(
    R_bar: float = 0.05,
    tension: float = 0.01,
    kb_scale: float = 60,
    osmolarity: float = 0.300,
    radial_subdivisions: int = 16,
    axial_subdivisions: int = 120,
    T: float = 310,
    reservoir_volume: float = 500,
    target_volume_scale: float = 3,
) -> Tuple[dg.Parameters, dg.Geometry, float]:
    """Initialize geometry and parameters for mem3dg simulation of axon pearling

    Args:
        R_bar (float, optional): Initial cylinder radius. Defaults to 0.05.
        tension (float, optional): Tension value in nN/μm. Defaults to 0.01.
        kb_scale (float, optional): Bending modulus in KT. Defaults to 60.
        osmolarity (float, optional): Osmolarity of extracellular solvent in mOsm. Defaults to 0.300.
        radial_subdivisions (int, optional): Number of radial subdivisions. Defaults to 16.
        axial_subdivisions (int, optional): Number of axial subdivisions for geometry. Defaults to 240.
        T (float, optional): temperature. Defaults to 310.
    Raises:
        RuntimeError: If R_bar, tension, bending modulus leads to a non-physical spontaneous curvature value.

    Returns:
        Tuple[dg.Parameters, dg.Geometry, float]:  Parameters, geometry, and suggested timestep
    """
    # Initialize constants and apply units
    R_bar = R_bar * unit.micrometer
    tension = (
        tension * unit.nanonewton / unit.micrometer
    )  # Membrane tension nN/um || mN/m

    T = T * unit.degK  # kelvin
    KT = (unit.boltzmann_constant * T).to(unit.nanonewton * unit.micrometer)
    Kb = kb_scale * KT  # Bending modulus
    # 60*KT is slightly less than 0.27 pNμm Hochmuth, Shao, Dai,Sheets
    # print("KB", Kb.to(unit.piconewton * unit.micrometer))

    # print(f"1/4R^2: \t{1 / (4 * R_bar * R_bar)}\nsigma/Kb:\t{tension / Kb}")

    # Spontaneous curvature of a tube given a target radius
    if 1 / (4 * R_bar * R_bar) < tension / Kb:
        raise RuntimeError(
            f"Invalid spontaneous curvature for {R_bar}, {tension}, {kb_scale}"
        )

    H0c = np.sqrt(1 / (4 * R_bar * R_bar) - tension / Kb)
    R_init = R_bar

    print(
        "-" * 50,
        "Parameters:",
        f"\tSpontaneous curvature: {H0c}",
        f"\tTension: {tension}",
        f"\tBending modulus: {kb_scale} KT = {Kb}",
        # From bar ziv paper
        f"\tInstability ratio: {Kb/(tension*R_bar*R_bar)} < 2/3?",
        "-" * 50,
        sep="\n",
    )

    # Generate a cylinder
    Face, Vertex = dg.getCylinder(
        radius=R_init.magnitude,
        radialSubdivision=radial_subdivisions,
        axialSubdivision=axial_subdivisions,
        frequency=1,
        amplitude=0,
    )

    cylinder_height = (np.max(Vertex[:, 2]) - np.min(Vertex[:, 2])) * unit.micrometer

    geometry = dg.Geometry(
        faceMatrix=Face,
        vertexMatrix=Vertex,
    )

    # Theoretical initial volume
    initial_volume = np.pi * R_init * R_init * cylinder_height

    print(
        "Geometric properties:",
        f"\tHeight: {cylinder_height}",
        f"\tInitial volume: {initial_volume}",
        sep="\n",
    )

    # Initialize parameters
    p = dg.Parameters()

    p.boundary.shapeBoundaryCondition = "fixed"
    p.boundary.proteinBoundaryCondition = "none"

    p.variation.isShapeVariation = True
    p.variation.isProteinVariation = False
    p.variation.isProteinConservation = False
    p.variation.geodesicMask = -1  # No geodesic mask

    # Set bending modulus to zero to allow uniform protein distribution to set bending modulus and spontaneous curvature
    p.bending.Kb = 0
    # Bending modulus
    p.bending.Kbc = Kb.magnitude
    # Spontaneous curvature
    p.bending.H0c = H0c.magnitude

    p.tension.form = partial(constantSurfaceTensionModel, tension=tension.magnitude)

    osmolarity = osmolarity * unit.molar
    cell_osmolarity = 0.300 * unit.molar
    RT = KT * unit.avogadro_constant  # L*kPa/mol*K
    # print("RT", RT, RT.to(unit.micrometer * unit.nanonewton / unit.attomole), RT.to(unit.liter * unit.kilopascal / unit.mole))

    # Reservoir volume?
    # https://www.nature.com/articles/s42003-021-02548-6
    reservoir_volume = reservoir_volume * unit.micron**3
    target_volume = target_volume_scale * initial_volume

    initial_solute = (cell_osmolarity * target_volume).to_base_units()

    print(
        "Osmotic Pressure:",
        f"\tInitial solute: {initial_solute.to(unit.attomole)}",
        f"\tConcentration: {(initial_solute / initial_volume).to(unit.molar)}",
        f"\tScaled Kv strength (iRTn): {(RT*initial_solute/reservoir_volume.magnitude).to(unit.micron*unit.nanonewton)}",
        f"\tTarget osmolarity: {osmolarity}",
        sep="\n",
    )

    p.osmotic.form = partial(
        ambientSolutionOsmoticPressureModelWUnit,
        RT=RT.to(unit.micrometer * unit.nanonewton / unit.attomole),
        ambient_concentration=osmolarity.to(unit.attomole / unit.micron**3),
        enclosed_solute=initial_solute.to(unit.attomole),
        reservoir_volume=reservoir_volume,
    )

    # delta_p = RT * (cell_osmolarity - osmolarity)
    # print(
    #     f"Delta osmotic pressure:  {delta_p.to(unit.nanonewton /unit.micron**2)}",
    #     f"Initial (pressure, energy): {p.osmotic.form(initial_volume.magnitude)}",
    #     sep="\n",
    # )

    predicted_timestep = 5e-3 * R_bar * R_bar / Kb

    return p, geometry, predicted_timestep


def initialize_mesh_processor_settings(system: dg.System, R_bar: float):
    """Set system mesh processor settings inplace

    Args:
        system (dg.System): System to update
        geometry (dg.Geometry): The current geometry
        R_bar (float): _description_
    """
    system.meshProcessor.meshMutator.mutateMeshPeriod = 100
    system.meshProcessor.meshMutator.isShiftVertex = True
    system.meshProcessor.meshMutator.flipNonDelaunay = True

    system.meshProcessor.meshMutator.minimumEdgeLength = 0.0001 * R_bar
    system.meshProcessor.meshMutator.maximumEdgeLength = 0.1

    print(
        f"Edge Length:",
        f"\tmin: {system.meshProcessor.meshMutator.minimumEdgeLength}",
        f"\tmean: {system.getGeometry().getEdgeLengths().mean()}",
        f"\tmax: {system.meshProcessor.meshMutator.maximumEdgeLength}",
        sep="\n",
    )

    # SPLIT OPERATION TOGGLES
    # Curvature tolerance and split curved aren't used since they cause trouble in this example
    system.meshProcessor.meshMutator.curvTol = R_bar / 50
    system.meshProcessor.meshMutator.splitCurved = False
    # print("curvatureTol", g.meshProcessor.meshMutator.curvTol)

    system.meshProcessor.meshMutator.targetFaceArea = (
        system.getGeometry().getFaceAreas().mean()
    )
    print(f"Target face area: {system.meshProcessor.meshMutator.targetFaceArea}")

    system.meshProcessor.meshMutator.splitLarge = True
    system.meshProcessor.meshMutator.splitLong = True
    system.meshProcessor.meshMutator.splitFat = True
    system.meshProcessor.meshMutator.splitSkinnyDelaunay = True

    # COLLAPSE OPERATION TOGGLES
    system.meshProcessor.meshMutator.collapseFlat = False
    system.meshProcessor.meshMutator.collapseSkinny = True
    system.meshProcessor.meshMutator.collapseSmall = True

    # Extra smoothing on vertices with an outlier bending energy
    system.meshProcessor.meshMutator.isSmoothenMesh = True


def run_simulation(args):
    (
        osmolarity,
        tension,
        kb_scale,
        target_volume_scale,
        reservoir_volume,
        output_dir,
    ) = args
    R_bar = 0.05
    print(
        f"Starting: {output_dir} - {osmolarity}, {tension}, {kb_scale}, {target_volume_scale}, {reservoir_volume}"
    )

    trajfile = output_dir / "traj.nc"
    if trajfile.exists():
        print(f"Ending {output_dir} trajfile exists")
        return

    with open(output_dir / "log.txt", "w") as fd:
        with contextlib.redirect_stdout(fd):
            try:
                parameters, geometry, h = getGeometryParameters(
                    R_bar=R_bar,
                    osmolarity=osmolarity,
                    tension=tension,
                    kb_scale=kb_scale,
                    target_volume_scale=target_volume_scale,
                    reservoir_volume=reservoir_volume,
                )
            except Exception as e:
                (output_dir / "log.txt").unlink()
                output_dir.rmdir()
                print(
                    f"Ending {output_dir} invalid spontaneous curvature",
                    file=sys.stderr,
                )
                return

            _shape = np.shape(geometry.getVertexMatrix())

            proteinDensity = np.full(_shape[0], 1)
            velocity = np.zeros(_shape)

            # print("Initial Volume", geometry.getVolume())

            system = dg.System(
                geometry=geometry,
                parameters=parameters,
                velocity=velocity,
                proteinDensity=proteinDensity,
            )

            initialize_mesh_processor_settings(system, R_bar)

            system.initialize()

            dt = h.magnitude
            print("Characteristic timestep:", dt)
            fe = dg.Euler(
                system=system,
                characteristicTimeStep=dt,
                totalTime=1e7 * dt,
                savePeriod=2e3 * dt,
                tolerance=1e-9,
                outputDirectory=str(output_dir),
                frame=0,
            )

            # Enable backtracking to find stable timestep
            fe.isBacktrack = True
            # Do not adapt timestep wrt to mesh size
            fe.ifAdaptiveStep = False
            # Whether to print status to stdout
            fe.ifPrintToConsole = True
            # Output a netcdf trajfile
            fe.ifOutputTrajFile = True
            # Do not output intermediate ply files
            fe.ifOutputMeshFile = False
            # print("\n\n\n")

            fd.flush()
            try:
                # dg.startProfiler("profile.prof")
                fe.integrate()
                # dg.stopProfiler()
            except Exception as e:
                print(e)

            # Save the last frame to ply file no matter what
            fe.saveData(
                ifOutputTrajFile=False, ifOutputMeshFile=True, ifPrintToConsole=False
            )
    print(f"Ending: {output_dir}")


if __name__ == "__main__":
    base_dir = Path("trajectories")
    base_dir.mkdir(exist_ok=True)

    args = []

    for i, osmolarity in enumerate(osmolarities):
        for j, tension in enumerate(tensions):
            for k, kappa in enumerate(bending_moduli):
                # print(i,j,k)
                output_dir = base_dir / Path(f"{i}_{j}_{k}")
                output_dir.mkdir(exist_ok=True)
                args.append(
                    (
                        osmolarity,
                        tension,
                        kappa,
                        target_volume_scale,
                        reservoir_volume,
                        output_dir,
                    )
                )

    r = process_map(run_simulation, args, max_workers=20)

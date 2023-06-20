import netCDF4 as nc

# import netcdf4 first to avoid strange error
# related: https://github.com/pydata/xarray/issues/7259

import pymem3dg as dg
import pymem3dg.util as dgu
import pymem3dg.visual as dgv
import pymem3dg.read.netcdf as dg_nc

import polyscope as ps

from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing import Tuple
from functools import partial

import pickle

from scipy.signal import argrelextrema

import warnings

from units import unit
from driver import getParameters

from tqdm.contrib.concurrent import process_map

import contextlib

import sys
import pdb


from operator import itemgetter

from plot_helper import *

from parameter_variation import (
    osmolarities,
    tensions,
    bending_moduli,
    target_volume_scale,
    reservoir_volume,
)

fig_dir = Path("Figures")


def getRadii(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Zero out z columns
    r = dgu.rowwiseNorm(coords[:, :2]).reshape(-1, 1)
    z = coords[:, 2].reshape(-1, 1)
    res = np.hstack((r, z))
    return res


def log_parser(output_dir):
    n = -1
    logfile = output_dir / "log.txt"
    if logfile.exists():
        with open(logfile, "r") as fd:
            for line in fd:
                if line.startswith("t:"):
                    data = line.split(", ")
                    # t = float(data[0].split(": ")[1])
                    n = int(data[1].split(": ")[1])
                if line.startswith("<<<<<"):
                    # print(f"{output_dir}: {n}")
                    n -= 1
                    break
    return n


def plot_traj(args):
    (
        osmolarity,
        tension,
        kappa,
        target_volume_scale,
        reservoir_volume,
        output_dir,
    ) = itemgetter(
        "osmolarity",
        "tension",
        "kappa",
        "target_volume_scale",
        "reservoir_volume",
        "output_dir",
    )(
        args
    )
    trajfile = output_dir / "traj.nc"

    if trajfile.exists():
        # print(trajfile, trajfile.exists())
        ds = nc.Dataset(trajfile)

        dims = ds.groups["Trajectory"].dimensions
        vars = ds.groups["Trajectory"].variables

        n_frames = dims["frame"].size
        times = np.array(vars["time"])
        coords = np.array(vars["coordinates"])
        vel = np.array(vars["velocities"])

        # get coords from last frame
        rz = getRadii(coords[-1].reshape(-1, 3))

        xbins = 80
        n, _ = np.histogram(rz[:, 1], bins=xbins)
        sy, bin_edges = np.histogram(rz[:, 1], bins=xbins, weights=rz[:, 0])

        mean_z = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mean_r = sy / n

        min_indices = argrelextrema(mean_r, np.less)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot(mean_z, mean_r)
        ax.scatter(mean_z[min_indices], mean_r[min_indices], marker="x", c="r", s=15)
        ax.set_ylim(0, 0.2)

        fig.savefig(
            fig_dir / f"{output_dir.stem}_{osmolarity}_{tension}_{kappa}_lastframe.pdf",
            format="pdf",
        )

        fig.clear()
        plt.close(fig)

        bead_diameter = 2 * np.max(mean_r) * 1000  # nanometer
        bead_length = np.mean(np.diff(mean_z[min_indices])) * 1000  # nanometer

        print(f"{output_dir.stem}: {int(bead_diameter)}; {int(bead_length)}")
        # print("\n\n")

    else:
        # Some trajectories won't exist due to parameters violating spontaneous curvature formula
        pass


def plot_configuration(ax, output_dir):
    trajfile = output_dir / "traj.nc"

    print(output_dir.stem)
    # get frame of interest
    foi = log_parser(output_dir)

    if trajfile.exists():
        ds = nc.Dataset(trajfile)

        dims = ds.groups["Trajectory"].dimensions
        vars = ds.groups["Trajectory"].variables

        n_frames = dims["frame"].size
        times = np.array(vars["time"])
        coords = np.array(vars["coordinates"])
        vel = np.array(vars["velocities"])

        # get coords from frame of interest
        rz = getRadii(coords[foi].reshape(-1, 3))

        xbins = 80
        n, _ = np.histogram(rz[:, 1], bins=xbins)
        sy, bin_edges = np.histogram(rz[:, 1], bins=xbins, weights=rz[:, 0])

        mean_z = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mean_r = sy / n

        min_indices = argrelextrema(mean_r, np.less)[0]

        ax.plot(mean_z, mean_r)
        ax.scatter(
            mean_z[min_indices],
            mean_r[min_indices],
            marker="x",
            c="r",
            s=15,
        )
        ax.set_ylim(0, 0.2)

        bead_diameter = 2 * np.max(mean_r) * 1000

        if min_indices.shape[0] == 1:
            bead_length = -1
        else:
            bead_length = np.mean(np.diff(mean_z[min_indices])) * 1000

        ax.text(1.25, 0.175, f"D{int(bead_diameter)} L{int(bead_length)}")

        # print(f"{output_dir.stem}[{foi}]: {bead_diameter}; {bead_length}")
        # print(f"{output_dir.stem}[{foi}]: {int(bead_diameter)}; {int(bead_length)}")
        return foi, bead_diameter, bead_length
    else:
        # print(f"{output_dir.stem} DNE")
        return -1, -1, -1


def plot_array():
    values = {}

    for j, tension in enumerate(tensions):
        if j == 2:
            return
        elif j == 1:
            fig, axes = plt.subplots(
                len(osmolarities) - 2,
                len(bending_moduli) - 1,
                figsize=(20, 20),
                sharex=True,
                sharey=True,
            )
        else:
            fig, axes = plt.subplots(
                len(osmolarities) - 2,
                len(bending_moduli),
                figsize=(20, 20),
                sharex=True,
                sharey=True,
            )

        for i, osmolarity in enumerate(osmolarities[0:-2]):
            for k, kappa in enumerate(bending_moduli):
                if j == 1:
                    if k == 0:
                        continue
                    else:
                        ax = axes[i, k - 1]
                else:
                    ax = axes[i, k]

                output_dir = base_dir / Path(f"{i}_{j}_{k}")
                values[(i, j, k)] = plot_configuration(ax, output_dir)

                if j == 1 and k == 1:
                    ax.set_ylabel(f"Π ({int(osmolarity*1000)} mOsm)")
                elif k == 0:
                    ax.set_ylabel(f"Π ({int(osmolarity*1000)} mOsm)")

                if i == len(osmolarities) - 3:
                    ax.set_xlabel(f"κ ({kappa}KT)")

        fig.supxlabel("axial (μm)")
        fig.supylabel("radial (μm)")
        plt.tight_layout()
        fig.savefig(
            fig_dir / f"final_shapes_tension{j}.pdf",
            format="pdf",
        )

        fig.clear()
        plt.close(fig)

    with open("bead_properties.pkl", "wb") as handle:
        pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return values


def plot_osmolarity_trends(values):
    cmap = mpl.colormaps["viridis"]
    c = cmap(np.linspace(0, 1, len(osmolarities) - 2))

    j = 0
    tension = tensions[j]

    for k, kappa in enumerate(bending_moduli):
        print(k, kappa)
        # k = 3
        # kappa = bending_moduli[k]
        fig, ax = plt.subplots(
            figsize=(3, 3),
        )

        for i, osmolarity in enumerate(osmolarities[0:-2]):
            tup = (i, j, k)
            _, bead_diameter, bead_length = values[tup]
            print(tup, values[tup])
            ax.scatter(
                bead_length,
                bead_diameter,
                color=c[i],
                label=f"{int(osmolarity*1000)} mOsm",
            )

        ax.set_xlabel("NSB length (nm)")
        ax.set_ylabel("NSB width (nm)")

        ax.legend(loc="upper left")

        ax.set_xlim([250, 750])
        ax.set_ylim([150, 450])

        plt.tight_layout()
        fig.savefig(fig_dir / f"osmolarity_trend_T{j}_K{k}.pdf", format="pdf")

        fig.clear()
        plt.close(fig)


def plot_rigidity_trends(values):
    cmap = mpl.colormaps["viridis"]
    c = cmap(np.linspace(0, 1, len(bending_moduli)))

    j = 0
    tension = tensions[j]

    for i, osmolarity in enumerate(osmolarities):
        fig, ax = plt.subplots(
            figsize=(3, 3),
        )

        for k, kappa in enumerate(bending_moduli):
            print(k, kappa, tension, osmolarity)

            tup = (i, j, k)
            _, bead_diameter, bead_length = values[tup]
            print(tup, values[tup])
            ax.scatter(bead_length, bead_diameter, color=c[k], label=f"{int(kappa)} KT")

        ax.set_xlabel("NSB length (nm)")
        ax.set_ylabel("NSB width (nm)")

        # ax.set_xlim([250, 750])
        # ax.set_ylim([150, 450])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        fig.savefig(fig_dir / f"rigidity_trend_T{j}_O{i}.pdf", format="pdf")

        fig.clear()
        plt.close(fig)


def snapshot_generator(values):
    snapshot_dir = fig_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    for i, osmolarity in enumerate(osmolarities):
        for j, tension in enumerate(tensions):
            for k, kappa in enumerate(bending_moduli):
                output_dir = base_dir / Path(f"{i}_{j}_{k}")

                tup = (i, j, k)
                foi, _, _ = values[tup]

                trajfile = output_dir / "traj.nc"
                if trajfile.exists():
                    ps.init()
                    dgv.polyscopeStyle()
                    ds = nc.Dataset(trajfile)

                    dims = ds.groups["Trajectory"].dimensions
                    vars = ds.groups["Trajectory"].variables
                    time = dg_nc.getData(str(trajfile), foi, "Trajectory", "time", 1)
                    geometry = dg.Geometry(str(trajfile), foi)

                    vertex = geometry.getVertexMatrix()
                    face = geometry.getFaceMatrix()
                    psmesh = ps.register_surface_mesh(
                        "mesh",
                        vertex,
                        face,
                        color=[1, 1, 1],
                        edge_width=0.5,
                        edge_color=[0, 0, 0],
                        transparency=1,
                        smooth_shade=True,
                    )
                    dgv.setPolyscopePermutations(psmesh, face, vertex)

                    ps.reset_camera_to_home_view()
                    ps.look_at(
                        camera_location=[1, 0, 1], target=[0, 0, 1], fly_to=False
                    )
                    ps.set_length_scale(1.4)
                    ps.set_SSAA_factor(2)
                    ps.screenshot(
                        filename=str(
                            snapshot_dir
                            / f"{i}_{j}_{k}__{osmolarity}_{tension}_{kappa}.png"
                        ),
                        transparent_bg=True,
                    )

                    # ps.show()


if __name__ == "__main__":
    base_dir = Path("trajectories")
    base_dir.mkdir(exist_ok=True)

    values = plot_array()

    # with open("bead_properties.pkl", "rb") as handle:
    #     values = pickle.load(handle)

    plot_osmolarity_trends(values)
    plot_rigidity_trends(values)
    snapshot_generator(values)

    # args = []
    # for i, osmolarity in enumerate(osmolarities):
    #     for j, tension in enumerate(tensions):
    #         for k, kappa in enumerate(bending_moduli):
    #             # print(i,j,k)
    #             output_dir = base_dir / Path(f"{i}_{j}_{k}")
    #             output_dir.mkdir(exist_ok=True)
    #             args.append(
    #                 {
    #                     "osmolarity": osmolarity,
    #                     "tension": tension,
    #                     "kappa": kappa,
    #                     "target_volume_scale": target_volume_scale,
    #                     "reservoir_volume": reservoir_volume,
    #                     "output_dir": output_dir,
    #                 }
    #             )

    # r = process_map(plot_traj, args, max_workers=10)

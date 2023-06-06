import pymem3dg as dg
import pymem3dg.util as dgu

from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing import Tuple
from functools import partial


from scipy.signal import argrelextrema

import warnings

from units import unit
from driver import getParameters

from tqdm.contrib.concurrent import process_map

import contextlib

import sys
import pdb

import netCDF4 as nc


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

        bead_diameter = 2 * np.max(mean_r) * 1000
        bead_length = np.mean(np.diff(mean_z[min_indices])) * 1000

        print(f"{output_dir.stem}: {int(bead_diameter)}; {int(bead_length)}")
        # print("\n\n")

    else:
        # Some trajectories won't exist due to parameters violating spontaneous curvature formula
        pass


def plot_configuration(ax, output_dir):
    trajfile = output_dir / "traj.nc"
    # print(trajfile, trajfile.exists())
    if trajfile.exists():
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
        bead_length = np.mean(np.diff(mean_z[min_indices])) * 1000

        ax.text(1.25, 0.175, f"D{int(bead_diameter)} L{int(bead_length)}")

        print(f"{output_dir.stem}: {int(bead_diameter)}; {int(bead_length)}")
    else:
        print(f"{output_dir.stem} DNE")


def plot_array():
    for j, tension in enumerate(tensions):
        fig, axes = plt.subplots(
            len(osmolarities),
            len(bending_moduli),
            figsize=(20, 20),
            sharex=True,
            sharey=True,
        )

        for i, osmolarity in enumerate(osmolarities):
            for k, kappa in enumerate(bending_moduli):
                output_dir = base_dir / Path(f"{i}_{j}_{k}")
                ax = axes[i, k]
                plot_configuration(ax, output_dir)
                if k == 0:
                    ax.set_ylabel(f"Π ({int(osmolarity*1000)} mOsm)")

                if i == len(osmolarities) - 1:
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


if __name__ == "__main__":
    base_dir = Path("trajectories")
    base_dir.mkdir(exist_ok=True)

    plot_array()

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

from functools import partial
import matplotlib
matplotlib.use('agg')

import ase
import ase.data
import ase.io
import ase.spectrum.band_structure
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from io import StringIO

def _mpl_to_png_bytestring(fig):
    """This function uses Matplotlib's FigureCanvasAgg backend to convert a MPL
    figure into a PNG bytestring. The bytestring is not encoded in this step.
    """
    import io
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if isinstance(fig, plt.Axes):
        fig = fig.figure
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return output.getvalue()

def mpl_to_html(fig, **kwargs):
    """Take a figure and render it directly to HTML. A PNG is created, and
    then encoded into base64 and decoded back to UTF-8 so that it can be stored
    inside a <img> HTML tag.
    ``kwargs`` supports arbitrary HTML attributes. An example usage of kwargs:
    >>> render_mpl(fig, style='width:480px; height:auto;')
    """
    from markupsafe import Markup
    import base64

    bstring = _mpl_to_png_bytestring(fig)
    png = base64.b64encode(bstring).decode('utf8')
    options = ' '.join([f'{key}="{val}"' for key, val in kwargs.items()])

    return Markup(f'<img src="data:image/png;base64,{png}" {options}>')

def render_mpl(fig):
    """This function returns a png file from a Matplotlib figure or subplots
    object. It is designed to be at the bottom of an endpoint function; instead
    of returning HTML or ``render_template()``, you return this instead.
    """
    from flask import Response
    return Response(_mpl_to_png_bytestring(fig), mimetype='image/png')



@partial(jax.jit, static_argnames=("n_atoms",))
def hessian_k(kpt, graph, H, n_atoms: int):
    r"""Compute the Hessian matrix at a given k-point.

    .. math::

        \hat H_{ij}(\vec k) = \sum_a H_{0i,aj} e^{i \vec k \cdot (\vec x_{aj} - \vec x_{0i})}

    Args:
        kpt: k-point
        graph: extended graph
        H: Hessian matrix, computed with `predict_hessian_matrix`
    """
    r = graph.nodes.positions
    ph = jnp.exp(-1j * jnp.dot(r[:, None, :] - r[None, :, :], kpt))[:, None, :, None]
    a = graph.nodes.index_cell0
    i = jnp.arange(3)
    Hk = (
        jnp.zeros((n_atoms, 3, n_atoms, 3), dtype=ph.dtype)
        .at[jnp.ix_(a, i, a, i)]
        .add(ph * H)
    )
    return Hk.reshape((n_atoms * 3, n_atoms * 3))

@jax.jit
def dynamical_matrix(kpt, graph, H, masses):
    r"""Dynamical matrix at a given k-point.

    .. math::

        D_{ij}(\vec k) = \hat H_{ij}(\vec k) / \sqrt{m_i m_j}

    """
    Hk = hessian_k(kpt, graph, H, masses.size)
    Hk = Hk.reshape((masses.size, 3, masses.size, 3))

    iM = 1 / jnp.sqrt(masses)
    Hk = jnp.einsum("i,iujv,j->iujv", iM, Hk, iM)
    Hk = Hk.reshape((3 * masses.size, 3 * masses.size))
    return Hk


def plot_bands(
    atoms, 
    graph, 
    hessian, 
    npoints=1000,
):
    # create ase cell object
    cell = atoms.cell.array  # [3, 3]
    cell = ase.Atoms(cell=cell, pbc=True).cell

    masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]

    rec_vecs = 2 * np.pi * cell.reciprocal().real
    mp_band_path = cell.bandpath(npoints=npoints)

    all_kpts = mp_band_path.kpts @ rec_vecs
    all_eigs = []

    for kpt in tqdm(all_kpts):
        Dk = dynamical_matrix(kpt, graph, hessian, masses)
        Dk = np.asarray(Dk)
        all_eigs.append(np.sort(np.sqrt(np.linalg.eigh(Dk)[0])))

    # Replace NaN values with 0s in all_eigs
    all_eigs = [np.nan_to_num(eigs, nan=0.0) for eigs in all_eigs]

    all_eigs = np.stack(all_eigs)

    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

    all_eigs = all_eigs * np.sqrt(conv_const) * hbar / cm_inv

    print (all_eigs)

    bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

    fig = plt.figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    bs.plot(ax=ax, emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
    plt.ylabel("Phonon Frequency (cm$^{-1}$)")
    plt.tight_layout()

    return fig
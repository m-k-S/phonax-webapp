from functools import partial

import jax
import jax.numpy as jnp
import jraph

from events import event_emitter

import threading

def callback(value):
    print (value)
    event_emitter.emit('progress', {'data': int(value)})

@partial(jax.jit, static_argnums=(1,))
def predict_hessian_matrix(
    w,
    model,  # model(relative_vectors, species, senders, receivers) -> [num_nodes]
    graph: jraph.GraphsTuple,
) -> jax.Array:
    """To be used with hessian_k"""

    # socket_callback(5)

    def energy_fn(positions):
        vectors = positions[graph.receivers] - positions[graph.senders]
        node_energies = model(
            w, vectors, graph.nodes.species, graph.senders, graph.receivers
        )  # [n_nodes, ]

        node_energies = node_energies * graph.nodes.mask_primitive
        return jnp.sum(node_energies)

    basis = jnp.eye(
        graph.nodes.positions.size, dtype=graph.nodes.positions.dtype
    ).reshape(-1, *graph.nodes.positions.shape)

    N = len(basis)

    def body_fn(i, hessian):
        basis_index = jnp.int32(i / N * 100)
        jax.experimental.io_callback(callback, (), basis_index)
        # jax.pure_callback(socket_callback, jnp.int32, i)
        return hessian.at[i].set(
            jax.jvp(
                jax.grad(energy_fn),
                (graph.nodes.positions,),
                (basis[i],),
            )[1]
        )

    hessian = jnp.zeros(
        (graph.nodes.positions.size,) + graph.nodes.positions.shape,
        dtype=graph.nodes.positions.dtype,
    )

    hessian = jax.lax.fori_loop(0, len(basis), body_fn, hessian)
    return hessian.reshape(graph.nodes.positions.shape + graph.nodes.positions.shape)


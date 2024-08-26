import jax
import jax.numpy as jnp

def test_hessian(graph):
    def mock_model(graph, key=jax.random.PRNGKey(0)):
        # Generate random positive values using exponential distribution
        random_values = jax.random.exponential(
            key, 
            shape=(graph.nodes.positions.shape + graph.nodes.positions.shape)
        )
        
        return random_values

    hessian = mock_model(graph)
    return hessian
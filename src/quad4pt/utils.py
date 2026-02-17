def import_x64_jax():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        return jax
    except ImportError:
        raise ImportError("JAX is not installed. Please install JAX to use this feature.")

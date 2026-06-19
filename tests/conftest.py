def pytest_configure(config):
    """Register JAXMg-specific test markers."""
    config.addinivalue_line("markers", "cpu_only: tests that do not require GPUs")
    config.addinivalue_line("markers", "gpu: tests that require at least one GPU")
    config.addinivalue_line("markers", "mpmd: rank-per-GPU subprocess tests")
    config.addinivalue_line("markers", "slow: longer validation sweeps")

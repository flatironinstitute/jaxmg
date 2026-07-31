def pytest_configure(config):
    """Register JAXMg-specific test markers."""
    config.addinivalue_line("markers", "cpu_only: tests that do not require GPUs")
    config.addinivalue_line("markers", "gpu: tests that require at least one GPU")
    config.addinivalue_line("markers", "single_gpu: tests that require one GPU")
    config.addinivalue_line(
        "markers", "multi_gpu: rank-per-GPU tests that require multiple GPUs"
    )
    config.addinivalue_line("markers", "slow: longer validation sweeps")

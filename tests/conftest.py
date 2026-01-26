def pytest_collection_modifyitems(config, items):
    selected = []
    deselected = []
    for item in items:
        selected.append(item)
        if "mpmd/test_launch_2d_grid_reduce.py" in item.nodeid:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected

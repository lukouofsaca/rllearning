_METHOD_REGISTRY = {}

def register_method(name: str):
    def decorator(cls):
        key = name.lower()
        if key in _METHOD_REGISTRY:
            raise KeyError(f"Method '{name}' already registered")
        _METHOD_REGISTRY[key] = cls
        return cls
    return decorator


def get_method(name: str):
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown method name: {name}")
    return _METHOD_REGISTRY[key]
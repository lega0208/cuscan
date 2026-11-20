"""Explicit, version-checked upstream reference configuration."""

from importlib import metadata

REFERENCE_VERSION = "0.6.0"
OPTIMIZERS = ("compatibility", "adam", "momentum")


def require_reference() -> None:
    """Reject environments that cannot reproduce the pinned reference."""
    for package, expected in (("umap-learn", REFERENCE_VERSION), ("pynndescent", "0.7.0")):
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Reference requires {package}=={expected}; found {actual}")


def reference_kwargs(*, optimizer: str, init: str, **overrides: object) -> dict[str, object]:
    """Build explicit modern-pipeline options without implicit backend selection."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown reference optimizer: {optimizer!r}")
    if init == "auto":
        raise ValueError("Reference initialization must be explicit, not 'auto'")
    if "compatibility_layout" in overrides:
        raise ValueError("Reference option 'compatibility_layout' cannot be overridden")

    return {
        **overrides,
        "optimizer": optimizer,
        "init": init,
        "compatibility_layout": False,
        "nearest_neighbor_backend": overrides.get("nearest_neighbor_backend", "pynndescent"),
    }

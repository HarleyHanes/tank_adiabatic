# Test-wide configuration
# - Force a non-interactive matplotlib backend so tests never pop GUI windows
# - Add other global test tweaks here as needed

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    # Matplotlib may not be installed for all environments; ignore if missing
    pass

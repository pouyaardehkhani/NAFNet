import importlib
import warnings

# List of packages to check
packages = [
    ("numpy", "np"),
    ("cv2", None),
    ("scipy", None),
    ("skimage", None),
]

def check_package(pkg_name):
    try:
        pkg = importlib.import_module(pkg_name)
        version = getattr(pkg, "__version__", "No version attribute")
        print(f"{pkg_name} : {version}")
    except ImportError:
        print(f"{pkg_name} : Not installed")

# Ignore warnings
warnings.filterwarnings("ignore")

print("=== Package Version Check ===")
for name, _ in packages:
    check_package(name)

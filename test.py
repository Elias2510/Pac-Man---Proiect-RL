import importlib

# Function to check and print the version of a library
def print_version(lib_name):
    try:
        lib = importlib.import_module(lib_name)
        print(f"{lib_name} version: {lib.__version__}")
    except ImportError:
        print(f"{lib_name} is not installed.")

# Check versions for required libraries
libraries = ["gym", "numpy", "stable_baselines3", "cv2"]
for lib in libraries:
    print_version(lib)
  

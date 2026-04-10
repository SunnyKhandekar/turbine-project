import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from turbine_project.dashboard import run_dashboard


run_dashboard(".")

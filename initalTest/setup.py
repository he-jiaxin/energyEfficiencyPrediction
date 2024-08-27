import os
import sys
from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

if sys.platform == "darwin":
    required.append("tensorflow-macos")
    required.append("tensorflow-metal")
else:
    required.append("tensorflow-gpu==2.3.0")

if __name__ == "__main__":
    use_scm_version = not os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
    setup(
        use_scm_version=use_scm_version,
        install_requires=required,
        test_suite="tests",
    )

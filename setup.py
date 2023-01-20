

#to build for pip
#CMAKE_ARGS="-DSyCL_Compiler=DPCPP -DSyCL_Compiler_BE=CUDA -DCMAKE_CXX_COMPILER=/media/nvme/sycl_workspace_arch/sycl_cpl/dpcpp/bin/clang++ -DCOMP_ROOT_DIR=/media/nvme/sycl_workspace_arch/sycl_cpl/dpcpp -DBUILD_TEST=true -DBUILD_SIM=true -DBUILD_PYLIB=true" CMAKE_BUILD_PARALLEL_LEVEL=36 pip install -v -e .



import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext



# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeBuildbotExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def is_ninja_available() -> bool:

    try:
        # pipe output to /dev/null for silence
        null = open("/dev/null", "w")
        subprocess.Popen(['ninja', '--version'], stdout=null, stderr=null)
        null.close()
        return True
    except OSError:
        return False


class CMakeBuildbotBuild(build_ext):
    def build_extension(self, ext: CMakeBuildbotExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"


        buildbot_args = []

        if(is_ninja_available()):
            buildbot_args += [
                "--gen", "ninja"
                ]
        else:
            buildbot_args += [
                "--gen", "make"
                ]


        buildbot_args += [
            "--build",
            "debug" if debug else "release"
            ]

        outdir = "build_pip"

        buildbot_args += [
            "--outdir",
            outdir
            ]

        buildbot_args += [
            "--lib"
            ]


        if "BUILDBOT_ARGS" in os.environ:
            buildbot_args += [item for item in os.environ["BUILDBOT_ARGS"].split(" ") if item]

        buildbot_args += [
            f"--cmakeargs=\"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep} -DPYTHON_EXECUTABLE={sys.executable}\"",  # not used on MSVC, but no harm
        ]

        

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["python3", ext.sourcedir+"/buildbot/configure.py"] + buildbot_args, check=True
        )
        subprocess.run(
            ["cmake", "--build", outdir], check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="shamrock",
    version="0.0.1",
    author="Timothée David--Cléris",
    author_email="timothee.david--cleris@ens-lyon.fr",
    description="SHAMROCK Code for astrophysics",
    long_description="",
    ext_modules=[CMakeBuildbotExtension("shamrock")],
    cmdclass={"build_ext": CMakeBuildbotBuild},
    zip_safe=False,
    extras_require={},
    python_requires=">=3.7",
)
@echo off

REM configuration of paths
set VS="<PATH_TO_vcvarsall.bat>"
set CPATH=<PATH_TO_NVIDIA_GPU_COMPUTING>\include

REM add tdm gcc stuff
set PATH=<PATH_TO_TDM_COMPILER>;%PATH%

REM configure path for msvc compilers
REM for a 32 bit installation change this line to
REM CALL %VS%\vcvarsall.bat
CALL %VS%\vcvarsall.bat amd64

REM return a shell
@echo on

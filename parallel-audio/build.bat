@echo off
rem Usage: build.bat            (default SAMPLE_RATE=44100, unchanged behavior)
rem        build.bat 192000     (overrides SAMPLE_RATE at compile time)
if not exist build mkdir build
echo Cleaning...
if exist build\spikeling_engine.exe del build\spikeling_engine.exe
echo Compiling with Optimization...
set SR_DEFINE=
if not "%~1"=="" set SR_DEFINE=-DSAMPLE_RATE=%~1
gcc -O3 -march=native -flto -fopenmp -Iinclude -Isrc %SR_DEFINE% -o build\spikeling_engine.exe src\spikeling_native.c src\miniaudio_impl.c src\kiss_fft.c -lole32 -lwinmm -lm
if %errorlevel% equ 0 (
    echo Build Successful!
    echo Launching with High Priority...
    start /high build\spikeling_engine.exe
) else (
    echo Build Failed!
)
pause
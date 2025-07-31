@echo off
set MATLAB=C:\Program Files\MATLAB\R2023a
"%MATLAB%\bin\win64\gmake" -f AC_Feeder_Control.mk  OPTS="-DTID01EQ=1"

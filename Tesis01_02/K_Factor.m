function [Gc, PhaseBoost, kfactor] = K_Factor(sys,fc,pm_deg)
%K_FACTOR Designs a controller using the K-Factor approach.
%   GC = K_FACTOR(SYS,FC,PM_DEG) returns the controller for SYS with a
%   open-loop bandwidth of FC herz and a phase margin of PM degrees.
%   Both GC and SYS are continuous-time transfer functions.
%
%   [GC, PHASEBOOST, KFACTOR] = K_FACTOR(...) also returns the phase
%   boosted and k-factor.
%
%   K-factor approach results in optimum zero, pole locations, for a given 
%   phase margin and cross over frequency. Controller can be designed
%   accurately for a given phase margin and cross over frequency.

%   Copyright 2007-2007 Xiaolin Mao. 
%   $Revision: 1.0 $  $Date: 2007/11/18 13:00 $

wc = 2*pi*fc;

[mag_sys, phase_sys] = bode(sys, wc);

PhaseBoost =  pm_deg - ((phase_sys - 90) + 180);

if PhaseBoost <= 0,
    % Type I controller
    Gc = tf(1,[1 0]);
elseif PhaseBoost < 90,
    % Type II controller (Integrator and Lead-Lag compensator): 
    % Gc = K(1+s/wz)/s(1+s/wp) 
    kfactor    =  tan(double((PhaseBoost+90)*pi/180)/2);
    wz         =  wc/kfactor;
    wp         =  wc*kfactor;
    Gc         =  tf([1/wz 1], [1/wp 1 0]);
else
    % Type III controller: Gc = K(1+s/wz)^2/s(1+s/wp)^2
    kfactor    =  tan(((PhaseBoost+180)*pi/180)/4);
    wz         =  wc/kfactor;
    wp         =  wc*kfactor;
    Gc         =  tf([1/wz^2 2/wz 1],[1/wp^2 2/wp 1 0]);
end

mag_wc     =  bode(sys*Gc, wc);
Gc         =  Gc/mag_wc;

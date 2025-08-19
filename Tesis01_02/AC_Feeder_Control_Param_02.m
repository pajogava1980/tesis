% Data for "AC_Feeder_Control.slx"

clear variables
clc

%% AC Grid Parameters

VLLrms=13.8e3;          % Grid rms line-to-line voltage in V
VLNrms=VLLrms/sqrt(3);  % Grid peak line-to-neutral voltage in V
VLNpk=VLNrms*sqrt(2);   % Grid peak line-to-neutral voltage in V

f1=60;                  % Grid frequency in Hz
T1=1/f1;                % Grid period in s
w1=2*pi*f1;             % Grid frequency in rad/s

Rl=0.1;                 % Line resistance in ohm
Ll=10e-3;               % Line inductance in H

R=10;                   % Load resistance in ohm
C=100e-6;               % Capacitor bank capacitance in F

%% Breaker Parameters

Ron=1e-3;               % Switch ON resistance in ohm
Rsnubber=10e6;          % Switch enubber resistance in ohm


%% PLL Parameters

Gp_pll=-tf(VLNpk,[1 0]);
BW_pll=100;
PM_pll=60;
Gc_pll=-K_Factor(-Gp_pll,BW_pll,PM_pll);

Z_pll=zero(Gc_pll);     % Controller zero
res = pole(Gc_pll);
P_pll=res(2);           % Controller pole
K_pll=dcgain(Gc_pll/tf([1 -Z_pll],[1 -P_pll 0])); % Controller DC gain


%% Simulation Parameters

Ts=50e-6;               % Simulation step time in s
Tsim=1;                 % Simulation end time in s

%% Create .mat Object

matObj=matfile('Register.mat','Writable',true);
%matObj.Vreg = [matObj.Vreg; Vreg]; % Añade los nuevos valores a Vreg
%disp('Datos guardados en Register.mat');
%matObj.Vreg(1,1) = 0;
%matObj.Vreg(2,1) = 0;

%% Interacting with Simulink

N=10;                   % Number of clock cycles
T=0.5;                  % Clk period in s
k=1;
Y=zeros(1,N);
t=zeros(1,N);
tap=0;
clk = logical(0);
%&set_param('AC_Feeder_Control', 'SimulationCommand', 'start')
%pause(T/2)


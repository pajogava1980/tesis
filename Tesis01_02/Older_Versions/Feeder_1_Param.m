% Parameters for "Feeder.slx"

% clear variables
% clc

%% Simulation Parameters
Ts=100e-6;

%% Create .mat File
% yo=1;
% save('myFile.mat','yo');

%% Create .mat Object
matObj=matfile('myFile.mat','Writable',true);

%% Load variable stored in .mat File
%y=matObj.yo;

%% Save variable stored in .mat File
% matObj.yo=y;%+1;

%% Identify 
%whos('-file','myFile.mat')

%% List current variables in .mat File
%load('myFile.mat','yo');

%% Interacting with Simulink
%get_param('Feeder/Ref','ObjectParameters')
%get_param('Feeder/Ref','Value')
a=20;
clk=false;
set_param('Feeder', 'SimulationCommand', 'start')
pause(0.5)
while a>0
    clk=~clk
    set_param('Feeder/Clk','Value','clk')
    if clk
        r=randi(100);
        %r=rand*100;
        set_param('Feeder/Ref','Value','r')
    end
    pause(0.5)
    n=size(matObj.yo);
    y=matObj.yo(2,n(1,2))
    a=a-1;
    if a==0
        set_param('Feeder', 'SimulationCommand', 'stop')
        break
    end
end

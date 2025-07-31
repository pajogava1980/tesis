% Parameters for "Feeder.slx"

clear variables
clc

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
a=10;
while a>0
    r=randi(100);
    %r=rand*100;
    set_param('Feeder/Ref','Value','r')
    n=length(matObj.yo);
    y=matObj.yo(2,n)
    pause(1)
    a=a-1;
    if a==0
        break
    end
end

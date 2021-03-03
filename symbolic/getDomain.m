function domain = getDomain()

addpath(genpath('C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\Utility'))
addpath('C:\Users\Sam\Documents\MATLAB\KolmogorovDefects\src')

options = struct;
options.Lx = 2*pi;  % width of the domain we are integrating over
options.Ly = 2*pi;  % height of the domain we are inegrating over
options.Ky = 4;     % wavenumber of forcing in the y direction
options.Nx = 128;   % spatial grid points along x
options.Ny = 128;   % spatial grid points along y
options.npu = 2^7;  % number of time steps per time unit
options.Lt = 1/30;   % number of (dimensionless) time units to integrate

domain = dom.KolmogorovDomainObject(options);
%endState = load("R40_turbulent_state_k1.mat","s");

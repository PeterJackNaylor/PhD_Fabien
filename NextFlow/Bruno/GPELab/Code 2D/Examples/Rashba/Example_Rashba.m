%%% This file is an example of how to use GPELab (FFT version)

%% GROUND STATE COMPUTATION WITH A ROTATING TERM AND COUPLED NONLINEARITIES


%-----------------------------------------------------------
% Setting the data
%-----------------------------------------------------------

%% Setting the method and geometry
Computation = 'Ground';
Ncomponents = 2;
Type = 'BESP';
Deltat = 1e-2;
Stop_time = [];
Stop_crit = {'MaxNorm',1e-6};
Method = Method_Var2d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
Method.Precond = 'FLaplace';
Method.Max_iter = 100;
xmin = -15;
xmax = 15;
ymin = -15;
ymax = 15;
Nx = 2^8+1;
Ny = 2^8+1;
Geometry2D = Geometry2D_Var2d(xmin,xmax,ymin,ymax,Nx,Ny);

%% Setting the physical problem
Delta = 0.5;
Beta = 1;
Beta_coupled = [5,4;4,5];
Kappa = 10;
Omega = 0.5;
Physics2D = Physics2D_Var2d(Method, Delta, Beta, Omega);
RashbaDispersion{1,1} = @(FFTX,FFTY) Delta*(FFTX.^2+FFTY.^2);
RashbaDispersion{1,2} = @(FFTX,FFTY) Kappa*FFTX - 1i*Kappa*FFTY;
RashbaDispersion{2,1} = @(FFTX,FFTY) Kappa*FFTX + 1i*Kappa*FFTY;
RashbaDispersion{2,2} = @(FFTX,FFTY) Delta*(FFTX.^2+FFTY.^2);
Physics2D = Dispersion_Var2d(Method, Physics2D,RashbaDispersion);
Physics2D = Potential_Var2d(Method, Physics2D);
Physics2D = Gradientx_Var2d(Method, Physics2D);
Physics2D = Gradienty_Var2d(Method, Physics2D);
Physics2D = Nonlinearity_Var2d(Method, Physics2D,Coupled_Cubic2d(Beta_coupled),...
[],Coupled_Cubic_energy2d(Beta_coupled));

%% Setting the initial data
InitialData_Choice = 2;
Phi_0 = InitialData_Var2d(Method, Geometry2D, Physics2D, InitialData_Choice);

%% Setting informations and outputs
Save = 0;
Outputs = OutputsINI_Var2d(Method, Save);
Printing = 1;
Evo = 10;
Draw = 1;
Print = Print_Var2d(Printing,Evo,Draw);

%-----------------------------------------------------------
% Launching simulation
%-----------------------------------------------------------

[Phi, Outputs] = GPELab2d(Phi_0,Method,Geometry2D,Physics2D,Outputs,[],Print);

%-----------------------------------------------------------
% Display the spin vector
%-----------------------------------------------------------

S = cell(3,1);
S{1} = @(Phi,X,Y) (conj(Phi{1}).*Phi{2} + conj(Phi{2}).*Phi{1})./(sqrt(abs(Phi{1}).^2+abs(Phi{2}).^2));
S{2} = @(Phi,X,Y) -1i*(conj(Phi{1}).*Phi{2} - conj(Phi{2}).*Phi{1})./(sqrt(abs(Phi{1}).^2+abs(Phi{2}).^2));
S{3} = @(Phi,X,Y) (abs(Phi{1}).^2 - abs(Phi{2}).^2)./(sqrt(abs(Phi{1}).^2+abs(Phi{2}).^2));
S_name = cell(3,1);
S_name{1} = 'Sx';
S_name{2} = 'Sy';
S_name{3} = 'Sz';
Draw_solution2d(Phi, Method, Geometry2D, Figure_Var2d, S, S_name);
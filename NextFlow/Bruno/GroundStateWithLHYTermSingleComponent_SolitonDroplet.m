function s=GroundStateWithLHYTermSingleComponent_SolitonDroplet(begining, ending)

close all
clear all

%%% This file is an example of how to use GPELab (FFT version)

%% GROUND STATE COMPUTATION WITH A LHY Term

% Setting the path
GPE_folder = 'GPELab'
addpath(genpath(GPE_folder))
DATFILE = 'scatteringlengthsimoninew.dat'   %'Y:\Theory\Cleaned Matlab script\Cleaned Matlab script\scatteringlengthsimoninew.dat'
SAVELOCATION = '/data/users/pnaylor/Bruno/Result/Phi_Up_N_'    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\Phi_Up_N_'
SAVEMAT = '/data/users/pnaylor/Bruno/Result/PhaseDiagram_nmax.mat'    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
SAVEMAT_Nat = '/data/users/pnaylor/Bruno/Result/PhaseDiagram_Nat.mat'    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
SAVEMAT_BVec = '/data/users/pnaylor/Bruno/Result/PhaseDiagram_BVec.mat'    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
%-----------------------------------------------------------
% Setting the data
%-----------------------------------------------------------

%% Setting the method and geometry
Computation = 'Ground';
Ncomponents = 1;
Type = 'BESP';
Deltat = 1e-4;
Stop_time = [];
Stop_crit = {'MaxNorm',1e-4};
Method = Method_Var3d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
xmin = -1.5;
xmax = 1.5;
ymin = -0.5;
ymax = 0.5;
zmin = -0.5;
zmax = 0.5;
Nx = 2^6+1;
Ny = 2^5+1;
Nz = 2^5+1;
Geometry3D = Geometry3D_Var3d(xmin,xmax,ymin,ymax,zmin,zmax,Nx,Ny,Nz);

%% Setting the physical problem

%%% Fundamental constants
a0=5.29e-11;
hbar=1.0545718e-34; 
m=39*1.66e-27;
pi=3.14159;

%%% Problem parameters

% define N grid
Npoint=41;
Nmax=6500;
Nmin=800;
DN=(Nmax-Nmin)/(Npoint-1);
%DN=100;
%Npoint=(Nmax-Nmin)/DN+1;
Nat=[Nmin:DN:Nmax];

% define B grid

Bpoint=100;
Bmin=55.1145;
Bmax=56.525;
DB=(Bmax-Bmin)/(Bpoint-1);
Bgrid=[Bmin:DB:Bmax];

BPointDiagram=32;
BminLoop=55.25;
BMaxLoop=56.0;
BStep=0.025;
BVec=[BminLoop:BStep:BMaxLoop];


%% get scattering from file
SimoniData = importdata(DATFILE);
BSimoni=SimoniData(:,1);
abbSimoni=SimoniData(:,2);
accSimoni=SimoniData(:,3);
abcSimoni=SimoniData(:,4);

abb = spline(BSimoni,abbSimoni,Bgrid)*a0;
abc = spline(BSimoni,abcSimoni,Bgrid)*a0;
acc = spline(BSimoni,accSimoni,Bgrid)*a0;


trapFrequencyRadial=109; %in Hz
trapFrequencyAxial=5; % in Hz (you have to put a non-zero frequency)



parfor j
% B=56.1;   
B=BminLoop+(j-1)*BStep;
[tt I]=min((B-Bgrid).^2);
a1=abb(I);
a2=acc(I),
a12=abc(I);

for i=1:Npoint
 

        



N=Nat(i);   % Total number of atoms

% a1=abb(i);
% a2=acc(i),
% a12=abc(i);

% 
Nm=2000+5500*(B-55.2).^2;
if N>Nm
Deltat=0.5e-3
else
Deltat=0.2e-2
end
Method = Method_Var3d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
%% calculation adimensioned parameters

%trap
omegam=2*pi*trapFrequencyAxial;
aHO=sqrt(hbar/m/2/pi/trapFrequencyAxial);
aHORadial=sqrt(hbar/m/2/pi/trapFrequencyRadial);

gamma_x=1;
gamma_y=trapFrequencyRadial/trapFrequencyAxial;
gamma_z=trapFrequencyRadial/trapFrequencyAxial;



% interactions
g1=a1*4*pi*hbar^2/m;g2=a2*4*pi*hbar^2/m;g12=a12*4*pi*hbar^2/m;
deltaa=a12+sqrt(a1*a2);
deltag=deltaa*4*pi*hbar^2/m;
Alpha=(1+sqrt(g1/g2))/2;
x=g12^2/g1/g2;
y=sqrt(g2/g1);
F=((1+y+sqrt((1-y)^2+4*x*y))^(5/2)+(abs(1+y-sqrt((1-y)^2+4*x*y)))^(5/2))/4/sqrt(2); %Petrov function

n0=N/(2*Alpha*aHO^3);
%% GP adimensioned parameters
beta=(2*Alpha-1)/(2*Alpha^2)*deltag/aHO^3/hbar/omegam*N; %mean field prefactor

gamma=8*F/(3*sqrt(2*pi)*Alpha^(5/2))*(g1/(aHO^3*hbar*omegam))*(a1/aHO)^(3/2)*N^(3/2); %LHY prefactor

%%% 
Physics3D = Physics3D_Var3d(Method, 0.5,1,0);
Physics3D = Dispersion_Var3d(Method, Physics3D);
Physics3D = Potential_Var3d(Method, Physics3D,@(X,Y,Z) quadratic_potential3d(gamma_x,gamma_y,gamma_z,X,Y,Z));
Physics3D = Nonlinearity_Var3d(Method, Physics3D,@(phi,x,y,z) beta*abs(phi).^2+gamma*abs(phi).^(3));
Physics3D = Gradientx_Var3d(Method, Physics3D);
Physics3D = Gradienty_Var3d(Method, Physics3D);

%% Setting the initial data
InitialData_Choice = 1;
if (i==1)
    
Phi_0 = InitialData_Var3d(Method, Geometry3D, Physics3D, InitialData_Choice);
%Phi_U=load(horzcat('D:\GPEWavefunctions\Phi_initial.mat'));
%Phi_0=Phi_U.Phi_0;  
else     
Phi_U=load(horzcat(SAVELOCATION, num2str(Nat(i-1)),'B_',num2str(B),'G.mat'));
Phi_0=Phi_U.Phi;  
end
%% Phi_U=load('C:\Users\pcheiney\Desktop\PhiDroplet');
%% Phi_0=Phi_U.Phi;
%% Setting informations and outputs
Outputs = OutputsINI_Var3d(Method);
Printing = 1;
Evo = 15;
Draw = 0;
Print = Print_Var2d(Printing,Evo,Draw);
%-----------------------------------------------------------
% Launching simulation
%-----------------------------------------------------------

[Phi, Outputs] = GPELab3d(Phi_0,Method,Geometry3D,Physics3D,Outputs,[],Print);
%Phisave(i)=Phi;

%%%save(horzcat('D:\GPEWavefunctions\Phi_Up_N_',num2str(Nat(i)),'B_',num2str(B),'G.mat'),'Phi');

save(horzcat(SAVELOCATION, num2str(Nat(i)),'B_',num2str(B),'G.mat'),'Phi');
%save(horzcat('Y:\Theory\Cleaned Matlab script\Cleaned Matlab script\Test\GPEWavefunctions\Phi_Up_N_',num2str(Nat),'B_',num2str(B),'G.mat'),'Phi');

%% Ploting ouptut

%%grid in um
X=[0:Nx-1]/(Nx-1)*(xmax-xmin)+xmin;X=X*aHO;
Y=[0:Ny-1]/(Ny-1)*(ymax-ymin)+ymin;Y=Y*aHO;
Z=[0:Nz-1]/(Nz-1)*(zmax-zmin)+zmin;Z=Z*aHO;
%%ThomasFermi profile

nTFmax=n0*0.5*(15/4/pi)^(2/5)*beta^(-3/5)*(gamma_x*gamma_y*gamma_z)^(2/5)
nTFX=max(0,nTFmax-0.5*n0*gamma_x^2/beta*(X/aHO).^2);
nTFY=max(0,nTFmax-0.5*n0*gamma_y^2/beta*(Y/aHO).^2);
nTFZ=max(0,nTFmax-0.5*n0*gamma_z^2/beta*(Z/aHO).^2);

dx=(xmax-xmin)/(Nx-1);
dy=(ymax-ymin)/(Ny-1);
dz=(zmax-zmin)/(Nz-1);

%%normalization is such that  sum(sum(sum(abs(Phi{1}).^2)))*dx*dy*dz=1

toPlot=0;
if (toPlot==1)
%%plot cuts along 3 directions
figure
%%% radial directions
subplot(3,1,2)
plot(Y*1e6,2*Alpha*n0*abs(squeeze(Phi{1}((end-1)/2,(end-1)/2,:))).^2,'--*');
fitY = fit(Y'*1e6,2*Alpha*n0*abs(squeeze(Phi{1}((end-1)/2,(end-1)/2,:))).^2,'gauss1')
hold on
plot(fitY)
%plot(Y*1e6,nTFY)
xlabel('y (um)')
ylabel('density (at/m^3)')
title(num2str(fitY.c1))

subplot(3,1,3)
plot(Z*1e6,2*Alpha*n0*abs(squeeze(Phi{1}(:,(end-1)/2,(end-1)/2))).^2,'--*');
fitZ = fit(Z'*1e6,2*Alpha*n0*abs(squeeze(Phi{1}(:,(end-1)/2,(end-1)/2))).^2,'gauss1')
hold on
plot(fitZ)
%plot(Z*1e6,nTFZ)
xlabel('z (um)')
ylabel('density (at/m^3)')
title(num2str(fitZ.c1))
%%% axial direction
subplot(3,1,1)
plot(X*1e6,2*Alpha*n0*abs(squeeze(Phi{1}((end-1)/2,:,(end-1)/2))).^2,'--*');
fitX = fit(X'*1e6,2*Alpha*n0*abs(squeeze(Phi{1}((end-1)/2,:,(end-1)/2))').^2,'gauss1')
hold on
plot(fitX)
%plot(X*1e6,nTFX)
xlabel('x (um)')
ylabel('density (at/m^3)')
title(num2str(fitX.c1))
end
%% calulate moments
[meshX,meshY,meshZ] = meshgrid(X,Y,Z);
nmax(j,i)=2*Alpha*n0*max(max(max(abs(Phi{1}).^2)))
COM(i).x=sum(sum(sum(meshX.*abs(Phi{1}).^2)));
COM(i).y=sum(sum(sum(meshX.*abs(Phi{1}).^2)));
COM(i).z=sum(sum(sum(meshX.*abs(Phi{1}).^2)));

variance(i).x=sum(sum(sum((meshX-COM(i).x).^2.*abs(Phi{1}).^2)));
variance(i).y=sum(sum(sum((meshY-COM(i).y).^2.*abs(Phi{1}).^2)));
variance(i).z=sum(sum(sum((meshZ-COM(i).z).^2.*abs(Phi{1}).^2)));
end


save(horzcat(SAVEMAT),'nmax');
save(horzcat(SAVEMAT_Nat), 'Nat');
save(horzcat(SAVEMAT_BVec), 'BVec');
s=1
end

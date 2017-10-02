function GroundStateWithLHYTermSingleComponent_SolitonDroplet(bpointdiagram, begining, ending)

    disp(begining)
    disp(ending)
    begining = str2num(begining)
    ending = str2num(ending)
    %%% This file is an example of how to use GPELab (FFT version)

    %% GROUND STATE COMPUTATION WITH A LHY Term

    % Setting the path
    GPE_folder = 'GPELab'
    addpath(genpath(GPE_folder))
    DATFILE = 'scatteringlengthsimoninew.dat'   %'Y:\Theory\Cleaned Matlab script\Cleaned Matlab script\scatteringlengthsimoninew.dat'
    SAVELOCATION = 'Phi_Up_N_'    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\Phi_Up_N_'
    SAVEMAT = horzcat('PhaseDiagram_nmax_', num2str(begining), '_', num2str(ending), '.mat')    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
    SAVEMAT_Nat = horzcat('PhaseDiagram_Nat.mat')    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
    SAVEMAT_BVec = horzcat('PhaseDiagram_BVec.mat')    %'Y:\Personal folders\Bruno\Soliton_Droplet\SolitonToDroplet\PhaseDiagram_nmax.mat'
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
    Geometry3D = Geometry3D_Var3d(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz);

    %% Setting the physical problem

    %%% Fundamental constants
    a0 = 5.29e-11;
    hbar = 1.0545718e-34; 
    m = 39 * 1.66e-27;
    pi = 3.14159;

    %%% Problem parameters

    % define N grid
    Npoint = 41;
    Nmax = 6500;
    Nmin = 800;
    DN = (Nmax - Nmin) / (Npoint - 1);
    Nat = [Nmin:DN:Nmax];

    % define B grid

    Bpoint = 10;
    Bmin = 55.1145;
    Bmax = 56.525;
    DB = (Bmax - Bmin) / (Bpoint - 1);
    Bgrid = [Bmin:DB:Bmax];

    BPointDiagram = str2num(bpointdiagram);
    BminLoop = 55.46447;
    BMaxLoop = 56.525;
    DBLoop = (BMaxLoop - BminLoop) / (BPointDiagram - 1);
    BVec = [BminLoop:DBLoop:BMaxLoop];


    %% get scattering from file
    SimoniData = importdata(DATFILE);
    BSimoni=SimoniData(:, 1);
    abbSimoni=SimoniData(:, 2);
    accSimoni=SimoniData(:, 3);
    abcSimoni=SimoniData(:, 4);

    abb = spline(BSimoni, abbSimoni, Bgrid) * a0;
    abc = spline(BSimoni, abcSimoni, Bgrid) * a0;
    acc = spline(BSimoni, accSimoni, Bgrid) * a0;


    trapFrequencyRadial = 109; %in Hz
    trapFrequencyAxial = 5; % in Hz (you have to put a non-zero frequency)


    parfor j = begining:ending
        % B=56.1;   
        B = BminLoop + (j - 1) * DBLoop;
        [tt I] = min((B - Bgrid).^2);
        a1 = abb(I);
        a2 = acc(I),
        a12 = abc(I);

        temp_nmax_j = zeros(Npoint)

        for i = Npoint:-1:1

            N = Nat(i);   % Total number of atoms
            Nm = 2000 + 5500 * (B - 55.2).^2;

            if N > Nm
                Deltat = 0.5e-3
            else
                Deltat = 0.2e-2
            end %end if

            Method = Method_Var3d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
            %% calculation adimensioned parameters
            %trap
            omegam = 2 * pi * trapFrequencyAxial;
            aHO = sqrt(hbar / m / 2 / pi / trapFrequencyAxial);
            aHORadial = sqrt(hbar / m / 2 / pi / trapFrequencyRadial);

            gamma_x = 1;
            gamma_y = trapFrequencyRadial / trapFrequencyAxial;
            gamma_z = trapFrequencyRadial / trapFrequencyAxial;

            % interactions
            g1 = a1 * 4 * pi * hbar^2 / m; 
            g2 = a2 * 4 * pi * hbar^2 / m; 
            g12 = a12 * 4 * pi * hbar^2 / m;
            deltaa = a12 + sqrt(a1 * a2);
            deltag = deltaa * 4 * pi * hbar^2 / m;
            Alpha = (1 + sqrt(g1 / g2)) / 2;
            x = g12^2 / g1 / g2;
            y = sqrt(g2 / g1);
            F = ((1 + y + sqrt(( 1 - y )^2 + 4 * x * y ))^(5/2) + (abs( 1 + y - sqrt(( 1 - y )^2 + 4 * x * y)))^(5/2)) / 4 / sqrt(2); %Petrov function
            n0 = N / (2 * Alpha * aHO^3);
            %% GP adimensioned parameters
            beta = (2 * Alpha - 1) / (2 * Alpha^2) * deltag / aHO^3 / hbar / omegam * N; %mean field prefactor
            gamma = 8 * F / (3 * sqrt(2 * pi) * Alpha^(5/2)) * (g1 / (aHO^3 * hbar * omegam)) * (a1 / aHO)^(3/2) * N^(3/2); %LHY prefactor

            %%% 
            Physics3D = Physics3D_Var3d(Method, 0.5, 1, 0);
            Physics3D = Dispersion_Var3d(Method, Physics3D);
            Physics3D = Potential_Var3d(Method, Physics3D, @(X, Y, Z) quadratic_potential3d(gamma_x, gamma_y, gamma_z, X, Y, Z));
            Physics3D = Nonlinearity_Var3d(Method, Physics3D, @(phi, x, y, z) beta * abs(phi).^2 + gamma * abs(phi).^(3));
            Physics3D = Gradientx_Var3d(Method, Physics3D);
            Physics3D = Gradienty_Var3d(Method, Physics3D);

            %% Setting the initial data
            InitialData_Choice = 1;

            if (i == Npoint)
                Phi_0 = InitialData_Var3d(Method, Geometry3D, Physics3D, InitialData_Choice);
            else     
                Phi_U = PhiSaved;
                Phi_0=Phi_U.Phi;  
            end

            Outputs = OutputsINI_Var3d(Method);
            Printing = 1;
            Evo = 15;
            Draw = 0;
            Print = Print_Var2d(Printing, Evo, Draw);

            %-----------------------------------------------------------
            % Launching simulation
            %-----------------------------------------------------------

            [Phi, Outputs] = GPELab3d(Phi_0, Method, Geometry3D, Physics3D, Outputs, [], Print);
            PhiSaved = Phi;
            temp_nmax_j(i) =  2*Alpha*n0*max(max(max(abs(Phi{1}).^2)))
        end

        nmax(j,:) = temp_nmax_j
    end %end j parfor loop

    save(horzcat(SAVEMAT),'nmax');
    save(horzcat(SAVEMAT_Nat), 'Nat');
    save(horzcat(SAVEMAT_BVec), 'BVec');
end % end function

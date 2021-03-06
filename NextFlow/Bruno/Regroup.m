function Regroup(bpointdiagram, space)

    BPointDiagram = str2num(bpointdiagram);
    NPoint = 20;
    space = str2num(space);
    %% create result matrix of the good size
%    file_name = horzcat('PhaseDiagram_nmax_', num2str(BPointDiagram - space), '_', num2str(BPointDiagram),'.mat')
%    nmax = importdata(file_name)
    nmax = zeros(NPoint, BPointDiagram);
    for j=1:(space+1):(BPointDiagram-space)

        file_name = horzcat('PhaseDiagram_nmax_', num2str(j), '_', num2str(j+space),'.mat');
        disp(file_name);
        nmax_tmp = importdata(file_name);
        nmax(j:(j+space),:) = nmax_tmp(j:(j+space),:);
        %% extrac column and past it into result matrix
    end

    save('FinalMat.mat','nmax');

end

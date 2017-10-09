function Regroup(bpointdiagram, space)

    BPointDiagram = str2num(bpointdiagram);
    NPoint = 100
    space = str2num(space)
    %% create result matrix of the good size
%    file_name = horzcat('PhaseDiagram_nmax_', num2str(BPointDiagram - space), '_', num2str(BPointDiagram),'.mat')
%    nmax = importdata(file_name)
    nmax_final = zeros(NPoint, 100)
    for j=BPointDiagram:(space+1):(100-space)

        file_name = horzcat('PhaseDiagram_nmax_', num2str(j), '_', num2str(j+space),'.mat')
        nmax_tmp = importdata(file_name)
        nmax_final(j:(j+space),:) = nmax_tmp(j:(j+space),:)
        %% extrac column and past it into result matrix

    end

    save('FinalMat.mat','nmax');

end

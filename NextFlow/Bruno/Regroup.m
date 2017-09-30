function s=Regroup()
BPointDiagram=32;
%% create result matrix of the good size
file_name = horzcat('PhaseDiagram_nmax_', num2str(BPointDiagram),'.mat')
nmax = importdata(file_name)
for j=(BPointDiagram-1):-1:1

file_name = horzcat('PhaseDiagram_nmax_', num2str(j),'.mat')
nmax_tmp = importdata(file_name)
nmax(j,:) = nmax_tmp(j,:)
%% extrac column and past it into result matrix

end

save('FinalMat.mat','nmax');

end

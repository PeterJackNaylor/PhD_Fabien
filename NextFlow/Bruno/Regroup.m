function s=Regroup()
BPointDiagram=32;
%% create result matrix of the good size
file_name = horzcat('PhaseDiagram_nmax_', num2str(1),'.mat')
nmax = importdata(file_name)
for j=2:BPointDiagram

file_name = horzcat('PhaseDiagram_nmax_', num2str(j),'.mat')
nmax_tmp = importdata(file_name)
nmax = nmax + nmax_tmp
%% extrac column and past it into result matrix

end

save('FinalMat.mat','nmax');

end

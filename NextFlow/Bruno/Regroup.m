function s=Regroup()
BPointDiagram=32;
%% create result matrix of the good size

for j=1:BPointDiagram

file_name = horzcat('PhaseDiagram_nmax_', num2str(j),'.mat')
nmax_j(j) = load(file_name)
%% extrac column and past it into result matrix

end

save('FinalMat.mat','nmax');

end

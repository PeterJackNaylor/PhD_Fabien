function Normalize(FOLDER, which_norm, reference, tool_box);



    addpath(genpath(tool_box));
    
    TargetImage = imread(reference);
    RES = './ImageFolder';
    copyfile(FOLDER, RES);
    files = dir(horzcat(RES, '/Slide_*'));
    S = size(files);
    S = S(1,1);
    verbose = 0;
    
    for (j = 1:S)
        organ_name = files(j,1).name ;
        files_j = dir(horzcat(RES, '/', organ_name, '/*.png'));
        small_S = size(files_j);
        small_S = small_S(1,1);
        for (i = 1:small_S)
            file_name = files_j(i,1).name;
            source_img = horzcat(RES, '/', organ_name, '/', file_name);
            SourceImage = imread(source_img);
            if (which_norm == 'Macenko')
                [ Norm_source ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);
            else if (which_norm == 'RGBHist')
                [ Norm_source ] = Norm( SourceImage, TargetImage, 'RGBHist', verbose );
                end
            end
            
            imwrite(Norm_source, source_img);
        end

    end
end
   
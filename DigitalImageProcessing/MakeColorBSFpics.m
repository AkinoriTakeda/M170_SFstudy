clear all
close all

%- load images -%
workdir=''; % path to the directory holding original images
picType={'Face','House','Target'};
nPics=cell(1, max(size(picType)));
margin=16;

for n=1:max(size(picType))
    filefolder=picType{n};
    
    % change working directory
    cd(fullfile(workdir, filefolder))
    
    % get picture names
    newImgs=dir('*.tif');
    
    [row, col]=size(newImgs);
    nPics{1,n}=row; % for plot
    newImgsName=cell(row,col);
    for m=1:row
        newImgsName{m,col}=newImgs(m,col).name;
    end
    clear newImgs m

    % make image set
    newImgsSet=cell(row,col);
    for m=1:row
        pic=imread(newImgsName{m,col});
        newImgsSet{m,col}=pic;
        clear pic
    end
    
    if n==1
        ImgsName=newImgsName;
        ImgsSet=newImgsSet;
    else
        ImgsName=[ImgsName;newImgsName];
        ImgsSet=[ImgsSet;newImgsSet];
    end
    
    clear filefolder m row col newImgsName newImgsSet
end
clear n

disp('Loading images: finished!')


%% [Process 1] trimming margin
TrimmedImgsSet=cell(size(ImgsSet,1), size(ImgsSet,2));
for n=1:max(size(ImgsSet))
    pic=ImgsSet{n};
    picName=ImgsName{n};
    
    % trim margin
    if picName(1)~='T'
        pic=pic((margin+1):(size(pic,1)-margin), (margin+1):(size(pic,2)-margin), :);
    end
    
    TrimmedImgsSet{n}=pic;
    clear pic picName
end
clear n

disp('Trimming margins in normalized images: finished!')


%% [Process 2] matching mean luminance
% calculate mean luminance & contrast
nImgs=max(size(TrimmedImgsSet));
MeanLumi=0;
%MeanSD=0;
for n=1:nImgs
    pic=rgb2ycbcr(TrimmedImgsSet{n});
    graypic=pic(:,:,1);
    
    MeanLumi=MeanLumi+mean2(graypic);
    %MeanSD=MeanSD+std2(graypic);
    clear pic graypic
end
MeanLumi=MeanLumi/nImgs;
%MeanSD=MeanSD/nImgs;
clear n nImgs

% [memo]
% mean luminance: 105.3178
% mean contrast: 44.3765
%  (in 16-235 scale)


% make gray picture set from color pictures
grayImgsSet=cell(size(TrimmedImgsSet,1),size(TrimmedImgsSet,2));
for n=1:max(size(ImgsSet))
    pic=rgb2ycbcr(TrimmedImgsSet{n});
    graypic=double(pic(:,:,1));
    
    grayImgsSet{n}=graypic;
    clear pic graypic
end
clear n


% Match mean luminance
matchedGrayImgsSet=cell(size(TrimmedImgsSet,1),size(TrimmedImgsSet,2));
for n=1:max(size(grayImgsSet))
    pic=grayImgsSet{n};
    
    box=cell(1,1);
    box{1}=pic;
    picLumiEquate=lumMatch(box,[],[MeanLumi std2(pic)]);
    
    matchedpic=picLumiEquate{1};
    matchedGrayImgsSet{n}=matchedpic;
    clear pic box picLumiEquate matchedpic
end    
clear n

    
% reconstruct color pictures with corrected gray pictures
matchedImgsSet=cell(size(TrimmedImgsSet,1),size(TrimmedImgsSet,2));
for n=1:max(size(TrimmedImgsSet))
    pic=rgb2ycbcr(TrimmedImgsSet{n});
    graypic=matchedGrayImgsSet{n};
    
    pic(:,:,1)=graypic(:,:);
    pic_backed=ycbcr2rgb(pic);
    matchedImgsSet{n}=pic_backed;
    clear pic graypic pic_backed
end
clear MeanLumi MeanSD n 

disp('Matching luminances of margin-trimmed pictures: finished!')


%% save matched images
workdir=''; % path to the directory in which processed images will be saved
for n=1:max(size(matchedImgsSet))
    pic=matchedImgsSet{n};
    picName=ImgsName{n};
    
    if picName(1)=='F'
        filefolder=picType{1};
    elseif picName(1)=='H'
        filefolder=picType{2};
    else
        filefolder=picType{3};
    end
    
    if strcmp(fullfile(workdir, filefolder), pwd)==0
        cd(fullfile(workdir, filefolder))
    end
    
    imwrite(pic, picName)
    clear pic picName filefolder
end
clear n

disp('Saving matched images: finished!')
disp('     ---> All processes were finished!!')


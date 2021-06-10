clear all
close all

%- load images -%
workdir=''; % path to the directory holding original images
picType={'Face','House'};
nPics=cell(1, max(size(picType)));
margin=16;

for n=1:max(size(picType))
    filefolder=picType{n};
    
    % change working directory
    if strcmp(fullfile(workdir, filefolder), pwd)==0
        cd(fullfile(workdir, filefolder))
    end

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
    
    clear filefolder m row col newImgsName newImgsSet newGrayImgsSet newImgsMask
end
clear n

disp('Loading images: finished!')


%% make equiluminant pictures
MeanLumi=0.4130; % =105.3178/255

EquiluminantImgsSet=cell(size(ImgsSet,1),size(ImgsSet,2));
for n=1:max(size(ImgsSet))
    im=double(ImgsSet{n})/255;
    
    % trimming margin
    im=im((margin+1):(size(im,1)-margin), (margin+1):(size(im,2)-margin), :);
    template=ones(size(im,1), size(im,2))*MeanLumi;
    
    
    disp(sprintf('Processing %d/%d images...', n, max(size(ImgsSet))))
    %- repeat processing until all pixels have mean luminance value -%
    flag=1;
    nTrial=1;
    while flag~=0
        %- process -%
        pic=rgb2ycbcr(im);
        
        % [Note (added on June 4, 2021)]
        % In this script, YCbCr images are newly obtained in every iteration (line 70).
        % However, I now realize that this procedure is not essential because the Cb and Cr components
        % do never change, and thus that obtaining them once is sufficient.
        % Indeed, I confirmed that repeating the transformation between the RGB and YCbCr color spaces
        % and replacing the luminance dimension for the same image can also create identical products.
        
        pic(:,:,1)=template(:,:);
        pic_backed=uint8(round(ycbcr2rgb(pic)*255));
        
        % check result %
        num=0;
        graypic=double(rgb2gray(pic_backed));
        for m=1:size(graypic,1)
            for l=1:size(graypic,2)
                diff=104-graypic(m,l); % Please change '104' to your target luminance value
                template(m,l)=template(m,l)+(diff/255)*0.1;

                if diff~=0
                    num=num+1;
                end
                clear diff
            end
        end
        clear m l graypic

        disp(sprintf('Trial%d: %d', nTrial, num))
        
        if num==0
            flag=0;
        end
        nTrial=nTrial+1;
    end
    
    
    EquiluminantImgsSet{n}=pic_backed;
    
    stim=rgb2gray(pic_backed);
    msg=sprintf('[%s] Min: %d, Max: %d, Mean: %0.4f, SD: %0.4f', ImgsName{n}(1:(max(size(ImgsName{n}))-8)), min(stim(:)), max(stim(:)), mean2(stim), std2(stim));
    disp(msg)
   
    clear im pic pic_backed msg stim flag nTrial num template
end
clear n MeanLumi


%% save modified images
workdir=''; % path to the directory in which modified images will be saved
for n=1:max(size(EquiluminantImgsSet))
    pic=EquiluminantImgsSet{n};
    picName=ImgsName{n}(1:(max(size(ImgsName{n}))-8));
    
    if picName(1)=='F'
        filefolder=picType{1};
        newpicName=[picName(1:3), picName(5),'Equ'];
    else
        filefolder=picType{2};
        newpicName=[picName(1:3), 'Equ'];
    end
    
    if strcmp(fullfile(workdir, filefolder), pwd)==0
        cd(fullfile(workdir, filefolder))
    end
    
    imwrite(pic, [newpicName, '.tif'])
    clear pic picName newpicName filefolder
end
clear n

disp('Saving equiluminant images: finished!')
disp('     ---> All processes were finished!!')


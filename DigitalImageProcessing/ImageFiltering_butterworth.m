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
    newGrayImgsSet=cell(row,col);
    newImgsMask=cell(row,col);
    for m=1:row
        pic=imread(newImgsName{m,col});
        picTransd=rgb2ycbcr(pic);
        graypic=picTransd(:,:,1);
        
        % make image mask for luminance adjustment
        mask=zeros(size(graypic,1),size(graypic,2));
        mask((margin+1):(size(graypic,1)-margin),(margin+1):(size(graypic,2)-margin))=1;
        
        newImgsSet{m,col}=pic;
        newGrayImgsSet{m,col}=graypic;
        newImgsMask{m,col}=mask;
        
        clear pic picTransd graypic mask
    end
    
    if n==1
        ImgsName=newImgsName;
        ImgsSet=newImgsSet;
        GrayImgsSet=newGrayImgsSet;
        ImgsMask=newImgsMask;
    else
        ImgsName=[ImgsName;newImgsName];
        ImgsSet=[ImgsSet;newImgsSet];
        GrayImgsSet=[GrayImgsSet;newGrayImgsSet];
        ImgsMask=[ImgsMask;newImgsMask];
    end
    
    clear filefolder m row col newImgsName newImgsSet newGrayImgsSet newImgsMask
end
clear n

disp('Loading images: finished!')


%% design 2D low-/high-pass butterworth filters 
[h, w]=size(GrayImgsSet{1});
[x, y]=meshgrid(-floor(w/2):floor(w/2)-1,-floor(h/2):floor(h/2)-1);

n=2; % n: order of the filter
B=1; % B: a controlling scale factor
D=sqrt(x.^2+y.^2);

% high-pass for HSF %
Hd=24; % d: cutoff distance
Highpass=1./(1+B*((Hd./D).^(2*n)));

% low-pass for LSF %
Ld=6; % d: cutoff distance
hhp=1./(1+B*((Ld./D).^(2*n)));
Lowpass=1.0-hhp;

clear h w x y n B D Hd Ld hhp
disp('Designing 2D low-/high-pass butterworth filters: finished!')


%% 2D-filtering grayscale images
Lowpassed_ImgsSet=cell(max(size(GrayImgsSet)),1);
Highpassed_ImgsSet=cell(max(size(GrayImgsSet)),1);

MeanLumi=105.3178; % mean luminance of chromatic BSF images in YCbCr color space
MaxYcbcrVal=235;
MinYcbcrVal=16;

tic
disp('Start filtering images.')
for n=1:max(size(GrayImgsSet))
    pic=GrayImgsSet{n};
    mask=ImgsMask{n};
    fftpic=fftshift(fft2(double(pic)));
   
    %- Low-pass filtering -%
    LowpassedPicFFT=fftpic.*Lowpass;
    LowpassedPic=real(ifft2(ifftshift(LowpassedPicFFT)));
    
    % normalize & rescale (0-219)
    LowpassedPic=((LowpassedPic-min(LowpassedPic(:)))/(max(LowpassedPic(:))-min(LowpassedPic(:))));
    LowpassedPic=LowpassedPic*(MaxYcbcrVal-MinYcbcrVal);
    
    % matching mean luminance
    box=cell(1,1);
    box{1}=LowpassedPic;
    LowLumiEquate=lumMatch(box, mask, [(MeanLumi-MinYcbcrVal), std2(LowpassedPic(mask==1))]);
    LowpassedPic=LowLumiEquate{1}+MinYcbcrVal;
    clear box LowLumiEquate
    
    stim=LowpassedPic(mask==1);
    msg=sprintf('[Low-passed %s] Min: %d, Max: %d, Mean: %0.1f, SD: %0.1f', ImgsName{n}(1:(max(size(ImgsName{n}))-8)), min(stim), max(stim), mean2(stim), std2(stim));
    disp(msg)
    msg2=sprintf('N of pixels below 16: %d (%0.1f %%)', size(stim(stim<MinYcbcrVal),1), (size(stim(stim<MinYcbcrVal),1)/max(size(stim(:))))*100);
    disp(msg2)
    msg3=sprintf('N of pixels above 235: %d (%0.1f %%)', size(stim(stim>MaxYcbcrVal),1), (size(stim(stim>MaxYcbcrVal),1)/max(size(stim(:))))*100);
    disp(msg3)
    clear msg msg2 msg3 stim

    
    %- High-pass filtering -%
    HighpassedPicFFT=fftpic.*Highpass;
    HighpassedPic=real(ifft2(ifftshift(HighpassedPicFFT)));
    
    % normalize & rescale (0-219)
    HighpassedPic=((HighpassedPic-min(HighpassedPic(:)))/(max(HighpassedPic(:))-min(HighpassedPic(:))));
    HighpassedPic=HighpassedPic*(MaxYcbcrVal-MinYcbcrVal);
    
    % matching mean luminance
    box=cell(1,1);
    box{1}=HighpassedPic;
    HighLumiEquate=lumMatch(box, mask, [(MeanLumi-MinYcbcrVal), std2(HighpassedPic(mask==1))]);
    HighpassedPic=HighLumiEquate{1}+MinYcbcrVal;
    clear box HighLumiEquate
    
    stim=HighpassedPic(mask==1);
    msg=sprintf('[High-passed %s] Min: %d, Max: %d, Mean: %0.1f, SD: %0.1f', ImgsName{n}(1:(max(size(ImgsName{n}))-8)), min(stim), max(stim), mean2(stim), std2(stim));
    disp(msg)
    msg2=sprintf('N of pixels below 16: %d (%0.1f %%)', size(stim(stim<MinYcbcrVal),1), (size(stim(stim<MinYcbcrVal),1)/max(size(stim(:))))*100);
    disp(msg2)
    msg3=sprintf('N of pixels above 235: %d (%0.1f %%)', size(stim(stim>MaxYcbcrVal),1), (size(stim(stim>MaxYcbcrVal),1)/max(size(stim(:))))*100);
    disp(msg3)
    clear msg msg2 msg3 stim

    
    %- save filtered images -%
    Lowpassed_ImgsSet{n,1} = LowpassedPic;
    Highpassed_ImgsSet{n,1} = HighpassedPic;
    
    progressMsg = sprintf('  %d/%d processes (%0.1f %%) finished...', n, max(size(ImgsSet)), (n/max(size(ImgsSet)))*100);
    disp(progressMsg)
    
    clear pic meanLumi fftpic LowpassedPic LowpassedPicFFT HighpassedPic HighpassedPicFFT progressMsg
end
clear n

disp('     ---> All filtering processes finished!!')
toc


%% reconstruct low-/high-passed color pictures
%- low-pass filtered images -%
Lowpassed_ColorImgsSet=cell(size(Lowpassed_ImgsSet,1),size(Lowpassed_ImgsSet,2));
for n=1:max(size(Lowpassed_ImgsSet))
    pic=rgb2ycbcr(ImgsSet{n});
    graypic=Lowpassed_ImgsSet{n};
    
    % reconstruct color picture
    pic(:,:,1)=graypic(:,:);
    pic_backed=ycbcr2rgb(pic);
    
    % trimming margin
    pic_backed=pic_backed((margin+1):(size(pic_backed,1)-margin), (margin+1):(size(pic_backed,2)-margin), :);
    Lowpassed_ColorImgsSet{n}=pic_backed;
    
    clear pic graypic pic_backed
end
clear n

%- high-pass filtered images -%
Highpassed_ColorImgsSet=cell(size(Highpassed_ImgsSet,1),size(Highpassed_ImgsSet,2));
for n=1:max(size(Highpassed_ImgsSet))
    pic=rgb2ycbcr(ImgsSet{n});
    graypic=Highpassed_ImgsSet{n};
    
    % reconstruct color picture
    pic(:,:,1)=graypic(:,:);
    pic_backed=ycbcr2rgb(pic);
    
    % trimming margin
    pic_backed=pic_backed((margin+1):(size(pic_backed,1)-margin), (margin+1):(size(pic_backed,2)-margin), :);
    Highpassed_ColorImgsSet{n}=pic_backed;
    
    clear pic graypic pic_backed
end
clear n


disp('Reconstructing filtered color images: finished!')


%% save modified images
workdir=''; % path to the directory in which modified images will be saved

%- 2. low-pass filtered & margin-trimmed images -%
for n=1:max(size(Lowpassed_ColorImgsSet))
    pic=Lowpassed_ColorImgsSet{n};
    picName=ImgsName{n}(1:(max(size(ImgsName{n}))-8));
    
    if picName(1)=='F'
        filefolder=picType{1};
        newpicName=[picName(1:3), picName(5)];
    else
        filefolder=picType{2};
        newpicName=picName(1:3);
    end
    
    if strcmp(fullfile(workdir, filefolder), pwd)==0
        cd(fullfile(workdir, filefolder))
    end
    
    imwrite(pic, [newpicName, 'LSF.tif'])
    clear pic picName newpicName filefolder
end
clear n

disp('Saving low-pass filtered color images: finished!')


%- 4. High-pass filtered & margin-trimmed images -%
for n=1:max(size(Highpassed_ColorImgsSet))
    pic=Highpassed_ColorImgsSet{n};
    picName=ImgsName{n}(1:(max(size(ImgsName{n}))-8));
    
    if picName(1)=='F'
        filefolder=picType{1};
        newpicName=[picName(1:3), picName(5)];
    else
        filefolder=picType{2};
        newpicName=picName(1:3);
    end
    
    if strcmp(fullfile(workdir, filefolder), pwd)==0
        cd(fullfile(workdir, filefolder))
    end
    
    imwrite(pic, [newpicName, 'HSF.tif'])
    clear pic picName newpicName filefolder
end
clear n

disp('Saving high-pass filtered color images: finished!')
disp('     ---> All processes were finished!!')


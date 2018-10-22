clear;
close all;
folder = 'datasets';

savepath = 'train.h5';

size_label = 96;
scale = 8;
size_input = size_label/scale;
size_x2 = size_label/4;
size_x4 = size_label/2;
stride = 48;
%% downsizing
downsizes = [1,0.7,0.5];

data = zeros(size_input, size_input, 1, 1);
label_x2 = zeros(size_x2, size_x2, 1, 1);
label_x4 = zeros(size_x4, size_x4, 1, 1);
label_x8 = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];


length(filepaths)

tic
for i = 1 : length(filepaths)
    for downsize = 1 : length(downsizes)
        image = imread(fullfile(folder,filepaths(i).name));
        image = imresize(image,downsizes(downsize),'bicubic');
        
        for flip = 1: 3
            if flip == 1
                image = flipdim(image ,1);
            end
            if flip == 2
                image = flipdim(image ,2);
            end
            
            for degree = 1 : 4
                image = imrotate(image, 90 * (degree - 1));
                if size(image,3)==3
                    image = rgb2ycbcr(image);
                    image = im2double(image(:, :, 1));
                    im_label = modcrop(image, scale);
                    [hei,wid] = size(im_label);

                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                            subim_label_x2 = imresize(subim_label,1/2,'bicubic');
                            subim_label_x4 = imresize(subim_label,1/4,'bicubic');
                            subim_input = imresize(subim_label,1/scale,'bicubic');

                            count=count+1;
                            data(:, :, :, count) = subim_input;            
                            label_x2(:, :, :, count) = subim_label_x4;  
                            label_x4(:, :, :, count) = subim_label_x2;   
                            label_x8(:, :, :, count) = subim_label;
                        end
                    end
                end
            end
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label_x8 = label_x8(:, :, :, order);
label_x4 = label_x4(:, :, :, order); 
label_x2 = label_x2(:, :, :, order);

disp({'Generate',int2str(count)})

%% writing to HDF5
chunksz = 256;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)

    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs_x2 = label_x2(:,:,:,last_read+1:last_read+chunksz);
    batchlabs_x4 = label_x4(:,:,:,last_read+1:last_read+chunksz);
    batchlabs_x8 = label_x8(:,:,:,last_read+1:last_read+chunksz);

    
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_x2', [1,1,1,totalct+1], 'lab_x4', [1,1,1,totalct+1], 'lab_x8', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_x2, batchlabs_x4, batchlabs_x8, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

toc
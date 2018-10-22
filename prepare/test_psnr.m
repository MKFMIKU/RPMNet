clear;close all;
%% settings
hr_folder = '/home/scw4750/Datasets/Set14';
sr_folder = '/home/scw4750/KFM/ICPR/SrSENet/ToTest';
scale = 2

hr_filepaths = dir(fullfile(hr_folder,'*.png'));
sr_filepaths = dir(fullfile(sr_folder,'*.bmp'));

num_img = length(hr_filepaths);

PSNR = zeros(num_img, 1);
SSIM = zeros(num_img, 1);

%% run
for i = 1:num_img
    fprintf('Testing on %s | %s \n', hr_filepaths(i).name, sr_filepaths(i).name);
    input_hr = fullfile(hr_folder, hr_filepaths(i).name);
    input_sr = fullfile(sr_folder, sr_filepaths(i).name);

    input_hr = im2double(imread(input_hr));
    input_sr = im2double(imread(input_sr));
    input_hr = mod_crop(input_hr, 4);

    if( size(input_hr, 3) > 1 )
        input_hr = rgb2ycbcr(input_hr); input_hr = input_hr(:, :, 1);
    end

    if( size(input_sr, 3) > 1 )
        input_sr = rgb2ycbcr(input_sr); input_sr = input_sr(:, :, 1);
    end


    input_hr = shave_bd(input_hr, scale);
    input_sr = shave_bd(input_sr, scale);

    PSNR(i) = psnr(input_hr, input_sr);
    SSIM(i) = ssim(input_hr, input_sr);

end

fprintf('Average PSNR = %f\n', mean(PSNR));
fprintf('Average SSIM = %f\n', mean(SSIM));


clear all;
close all;

data1 = load('wheel.mat');
data1 = data1.data;
data2 = load('wheel1.mat');
data2 = data2.data;
data3 = load('wheel2.mat');
data3 = data3.data;

data = zeros(18000,3,128,128);
data(1:6000,:,:,:) = data1;
data(6001:12000,:,:,:) = data2(6001:12000,:,:,:);
data(12001:18000,:,:,:) = data3(12001:18000,:,:,:);
data(:,3,:,:) = data(:,3,:,:)/12; % the maximum number of spoke is 12
clear data1 data2 data3;

for i = 1:18000
    aa = permute(squeeze(data(i,:,:,:)), [2,3,1]);
    aa(:,:,1) = aa(:,:,1)*255;
    aa(:,:,2) = round(aa(1,1,2)*255)*ones(128,128);
    aa(:,:,3) = round(aa(1,1,3)*255)*ones(128,128);
    aa = uint8(aa);
    figure();
    imshow(aa,[]);
    filename = strcat(num2str(i),'.png');
    imwrite(aa,filename,'png');
    close all;
end

% permutation
P = randperm(18000);
data_shuffle = data(P,:,:,:);
data = data_shuffle;
clear data_shuffle;

save('wheel_summary.mat', 'data', '-v7.3');

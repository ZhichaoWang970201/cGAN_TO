clear all;
close all;
clc;

data = zeros(6000,1,128,128); % 1: pixel; 2: vf; 3: number of spoke
num_im = 12001;
for num = 2001:3000
    % start with large resolution
    num_pix = 128;
    %r_i_l = (0.05+0.025*rand(1))*num_pix; % 0.05-0.075 of number of pixel 
    %r_i_r = (0.1+0.05*rand(1))*num_pix; % 0.1-0.15 of number of pixel
    %r_o_l = (0.375+0.05*rand(1))*num_pix; % 0.375-0.42 of number of pixel
    %r_o_r = 0.5*num_pix;
    r_i_l = 0.075 * num_pix;
    r_i_r = (0.1+0.05*rand(1))*num_pix;
    r_o_l = 0.4*num_pix;
    r_o_r = 0.5*num_pix;

    pix = zeros(num_pix,num_pix);
    for i = 1:num_pix
        for j = 1:num_pix
            dist = sqrt((i-num_pix/2-0.5)^2+(j-num_pix/2-0.5)^2);
            if dist>=r_i_l && dist<=r_i_r
                pix(i,j) = 1;
            end
            if dist>=r_o_l && dist<=r_o_r
                pix(i,j) = 1;
            end
        end
    end

    % Now, it is the time to add spoke
    num_spoke = randi([4,12]);
    theta_max = 2*pi/num_spoke;
    theta_interval = theta_max;

    % create inter angle
    theta_in = zeros(num_spoke,2);
    theta_in(1,:) = [ -pi , -pi+theta_interval ]; 
    for i = 2:num_spoke
        theta_in(i,:) = theta_in(i-1,:) + theta_max;
    end

    % create straight lines or inclined lines
    r1 = (r_i_l+r_i_r)/2;
    r3 = 0.5*num_pix;
    r2 = (r1+r3)/2;    
    
    % create outer angle
    theta_out = zeros(num_spoke,2);
    theta_interval_out = (rand(1)*0.4+1.4)*r1/r3*theta_interval;
    for i = 1:num_spoke
        theta_out(i,:) = [mean(theta_in(i,:))-theta_interval_out/2, mean(theta_in(i,:))+theta_interval_out/2];
    end
        
    % theta_mid
    theta_mid = zeros(num_spoke,2);
    theta_interval_mid = (rand(1)*0.2+1.2)*r1/r2*theta_interval;
    for i = 1:num_spoke
        theta_mid(i,:) = [mean(theta_in(i,:))-theta_interval_mid/2, mean(theta_in(i,:))+theta_interval_mid/2];
    end
    
    spoke_idx = randi([0 4]);
    if spoke_idx == 0
        r = rand(1);
        theta_mid = theta_mid + min(r*theta_max, pi/4);
    else
        rr1 = rand(1);
        rr2 = 0.5*rand(1)+0.5;
        theta_mid = theta_mid + min((rr1+rr2)*theta_max, pi/2);
        theta_out = theta_out + min(rr1*theta_max, pi/4);
        theta_in(:,2) = theta_in(:,2) + min(rr1*theta_max/2,pi/8);
    end
    
    u = 0:0.1:1;
    u1 = (1-u).^2;
    u2 = 2*u.*(1-u);
    u3 = u.^2;
        
    xx = zeros(22,num_spoke); % store x coordinates
    yy = zeros(22,num_spoke); % store y coordinates

    for i=1:num_spoke
        % the first part
        x = [r1*cos(theta_in(i,1)), r2*cos(theta_mid(i,1)), r3*cos(theta_out(i,1))]+num_pix/2+0.5;
        y = [r1*sin(theta_in(i,1)), r2*sin(theta_mid(i,1)), r3*sin(theta_out(i,1))]+num_pix/2+0.5;
        xx(1:11,i) = x(1)*u1 + x(2)*u2 + x(3)*u3;
        yy(1:11,i) = y(1)*u1 + y(2)*u2 + y(3)*u3;
        % the second part
        x = [r1*cos(theta_in(i,2)), r2*cos(theta_mid(i,2)), r3*cos(theta_out(i,2))]+num_pix/2+0.5;
        y = [r1*sin(theta_in(i,2)), r2*sin(theta_mid(i,2)), r3*sin(theta_out(i,2))]+num_pix/2+0.5;
        xx(12:22,i) = flip(x(1)*u1 + x(2)*u2 + x(3)*u3);
        yy(12:22,i) = flip(y(1)*u1 + y(2)*u2 + y(3)*u3);
    end
    
    for i = 1:num_spoke
        for j = 1:num_pix
            for k = 1:num_pix
                if inpolygon(j,k,xx(:,i),yy(:,i)) 
                    pix(j,k)=1;
                end
            end
        end
    end
    
    % add passive zone again
    for i = 1:num_pix
        for j = 1:num_pix
            dist = sqrt((i-num_pix/2-0.5)^2+(j-num_pix/2-0.5)^2);
            if dist>=r_i_l && dist<=r_i_r
                pix(i,j) = 1;
            end
            if dist>=r_o_l && dist<=r_o_r
                pix(i,j) = 1;
            end
            if dist<r_i_l
                pix(i,j) = 0;
            end
        end
    end

    % rotate and mirror at 5 different cases
    for ro_mi = 1:3
        rotate_theta = rand(1)*360;
        pix = imrotate(pix,rotate_theta,'bilinear','crop');

        for ii = 1:num_pix
            for jj = 1:num_pix
                if pix(ii,jj)>0.5
                    pix(ii,jj) = 1;
                else
                    pix(ii,jj) = 0;
                end
            end
        end
        
        filename = strcat(num2str(num_im),'.jpg');
        data(num_im,1,:,:) = pix;
        data(num_im,2,:,:) = sum(sum(pix))/128/128*ones(128,128);
        data(num_im,3,:,:) = num_spoke*ones(128,128);
        num_im = num_im + 1;
        imwrite(pix,filename);
    
        pix = flipdim(pix ,2);           %# horizontal flip
        for ii = 1:num_pix
            for jj = 1:num_pix
                if pix(ii,jj)>0.5
                    pix(ii,jj) = 1;
                else
                    pix(ii,jj) = 0;
                end
            end
        end
        
        filename = strcat(num2str(num_im),'.jpg');
        data(num_im,1,:,:) = pix;
        data(num_im,2,:,:) = sum(sum(pix))/128/128*ones(128,128);
        data(num_im,3,:,:) = num_spoke*ones(128,128);
        num_im = num_im + 1;
        imwrite(pix,filename);
    end
end
save('wheel2.mat','data', '-v7.3');
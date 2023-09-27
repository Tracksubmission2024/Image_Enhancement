% PSNR_vs_Alpha plot for 
clc;
clear;
close all;

% Input RGB
image_rgb = imread('Image1.jpg'); % Reading an RGB input image
image_rgb = double(image_rgb);
image_r = image_rgb(:,:,1); % Red Channel
image_g = image_rgb(:,:,2); % Green Channel
image_b = image_rgb(:,:,3); % Blue Channel
[M, N] = size(image_r); % Size of image
L = 256; % Number of gray levels
l = 0:L-1;

alpha_values = 0:0.01:1.5; % Alpha values from 0.5 to 1.5 with a step of 0.01
psnr_values = zeros(size(alpha_values)); % Array to store PSNR values

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    
     %energy curve
    B_r= zeros(M,N,L);
    B_g= zeros(M,N,L);
    B_b= zeros(M,N,L);
    for k= 0:L-1
        %Generation of Bg Matrix
        B_r(:,:,k+1)= 2*(image_r > k)-1;
        B_g(:,:,k+1)= 2*(image_g > k)-1;
        B_b(:,:,k+1)= 2*(image_b > k)-1;
    end
    N_d= ones(3); % 3*3 neighbourhood
    N_d((length(N_d)+1)/2,(length(N_d)+1)/2)= 0; %center element 0
    C= ones(M,N); %Generation of C matrix
    E_r= zeros(1,L);
    E_g= zeros(1,L);
    E_b= zeros(1,L);
    for k= 1:L
        %Energy Calculation at a particular intensity level
        E_r(k)= sum(sum(-1*B_r(:,:,k).*conv2(B_r(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
        E_g(k)= sum(sum(-1*B_g(:,:,k).*conv2(B_g(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
        E_b(k)= sum(sum(-1*B_b(:,:,k).*conv2(B_b(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
    end
    % clip
    C_clip_r = (median(E_r) + mean(E_r)) / 2; % Clipping limit of Red channel
    C_clip_g = (median(E_g) + mean(E_g)) / 2; % Clipping limit of Green channel
    C_clip_b = (median(E_b) + mean(E_b)) / 2; % Clipping limit of Blue channel
    
    E_hat_r= E_r;
    E_hat_g= E_g;
    E_hat_b= E_b;
    
    
    E_hat_r(E_hat_r >= C_clip_r) = C_clip_r; % Clipped Energy curve for Red channel
    E_hat_g(E_hat_g >= C_clip_g) = C_clip_g; % Clipped Energy curve for Green channel
    E_hat_b(E_hat_b >= C_clip_b) = C_clip_b; % Clipped Energy curve for Blue channel
    
    % standard deviation
    l_mean_r= sum(l.*E_hat_r)/sum(E_hat_r);
    l_mean_g= sum(l.*E_hat_g)/sum(E_hat_g);
    l_mean_b= sum(l.*E_hat_b)/sum(E_hat_b);
    SD_r= sqrt(sum(((l - l_mean_r).^2).*E_r)/sum(E_r));
    SD_g= sqrt(sum(((l - l_mean_g).^2).*E_g)/sum(E_g));
    SD_b= sqrt(sum(((l - l_mean_b).^2).*E_b)/sum(E_b));
    SD_r= round(SD_r);
    SD_g= round(SD_g);
    SD_b= round(SD_b);
    
    %partition
    L_hat_0_r= min(image_r,[],'all'); %Minimum intensity level for Red channel
    L_hat_L_1_r= max(image_r,[],'all'); %Maximum intensity level for Red channel
    L_low_r= L_hat_0_r + SD_r; %Lower limit for Red Channel
    L_high_r= L_hat_L_1_r - SD_r; %Upper limit for Red Channel
    L_hat_0_g= min(image_g,[],'all'); %Minimum intensity level for Green channel
    L_hat_L_1_g= max(image_g,[],'all'); %Maximum intensity level for Green channel
    L_low_g= L_hat_0_g + SD_g; %Lower limit for Green Channel
    L_high_g= L_hat_L_1_g - SD_g; %Upper limit for Green Channel
    L_hat_0_b= min(image_b,[],'all'); %Minimum intensity level for Blue channel
    L_hat_L_1_b= max(image_b,[],'all'); %Maximum intensity level for Blue channel
    L_low_b= L_hat_0_b + SD_b; %Lower limit for Blue Channel
    L_high_b= L_hat_L_1_b - SD_b; %Upper limit for Blue Channel
    
    
    % pdf and cdf calculation
    %PDF for sub-energy curves of Red Channel
    pdf_L_r= E_hat_r(1:L_low_r+1)/sum(E_hat_r(1:L_low_r+1));
    pdf_M_r= E_hat_r(L_low_r+2:L_high_r+1)/sum(E_hat_r(L_low_r+2:L_high_r+1));
    pdf_U_r= E_hat_r(L_high_r+2:L)/sum(E_hat_r(L_high_r+2:L));
    %CDF for sub-energy curves of Red Channel
    cdf_L_r= cumsum(pdf_L_r);
    cdf_M_r= cumsum(pdf_M_r);
    cdf_U_r= cumsum(pdf_U_r);
    %PDF for sub-energy curves of Green Channel
    pdf_L_g= E_hat_g(1:L_low_g+1)/sum(E_hat_g(1:L_low_g+1));
    pdf_M_g= E_hat_g(L_low_g+2:L_high_g+1)/sum(E_hat_g(L_low_g+2:L_high_g+1));
    pdf_U_g= E_hat_g(L_high_g+2:L)/sum(E_hat_g(L_high_g+2:L));
    %CDF for sub-energy curves of Green Channel
    cdf_L_g= cumsum(pdf_L_g);
    cdf_M_g= cumsum(pdf_M_g);
    cdf_U_g= cumsum(pdf_U_g);
    %PDF for sub-energy curves of Blue Channel
    pdf_L_b= E_hat_b(1:L_low_b+1)/sum(E_hat_b(1:L_low_b+1));
    pdf_M_b= E_hat_b(L_low_b+2:L_high_b+1)/sum(E_hat_b(L_low_b+2:L_high_b+1));
    pdf_U_b= E_hat_b(L_high_b+2:L)/sum(E_hat_b(L_high_b+2:L));
    %CDF for sub-energy curves of Blue Channel
    cdf_L_b= cumsum(pdf_L_b);
    cdf_M_b= cumsum(pdf_M_b);
    cdf_U_b= cumsum(pdf_U_b);
    
    % Adjust transfer functions for Red channel
    T_L_r = L_low_r * cdf_L_r;
    T_M_r = (L_low_r + 1) + alpha * (L_high_r - L_low_r + 1) * cdf_M_r;
    T_U_r = (L_high_r + 1) + alpha * (L - L_high_r + 1) * cdf_U_r;
    
    % Adjust transfer functions for Green channel
    T_L_g = L_low_g * cdf_L_g;
    T_M_g = (L_low_g + 1) + alpha * (L_high_g - L_low_g + 1) * cdf_M_g;
    T_U_g = (L_high_g + 1) + alpha * (L - L_high_g + 1) * cdf_U_g;
    
    % Adjust transfer functions for Blue channel
    T_L_b = L_low_b * cdf_L_b;
    T_M_b = (L_low_b + 1) + alpha * (L_high_b - L_low_b + 1) * cdf_M_b;
    T_U_b = (L_high_b + 1) + alpha * (L - L_high_b + 1) * cdf_U_b;
    
    % Final Transfer Function for Red channel
    T_r = [T_L_r, T_M_r, T_U_r];
    
    % Final Transfer Function for Green channel
    T_g = [T_L_g, T_M_g, T_U_g];
    
    % Final Transfer Function for Blue channel
    T_b = [T_L_b, T_M_b, T_U_b];
    
    
    %output RGB image
    image_out_r= image_r;
    image_out_g= image_g;
    image_out_b= image_b;
    %Applying transfer function
    for k= 0:L-1
        image_out_r(image_out_r== k) = T_r(k+1);
        image_out_g(image_out_g== k) = T_g(k+1);
        image_out_b(image_out_b== k) = T_b(k+1);
    end
    % output energy curve
    B_out_r= zeros(M,N,L);
    B_out_g= zeros(M,N,L);
    B_out_b= zeros(M,N,L);
    for k= 0:L-1
        %Generation of Bg Matrix for output RGB image
        B_out_r(:,:,k+1)= 2*(image_out_r > k)-1;
        B_out_g(:,:,k+1)= 2*(image_out_g > k)-1;
        B_out_b(:,:,k+1)= 2*(image_out_b > k)-1;
    end
    E_out_r= zeros(1,L);
    E_out_g= zeros(1,L);
    E_out_b= zeros(1,L);
    for k= 1:L
        %Energy Calculation
        E_out_r(k)= sum(sum(-1*B_out_r(:,:,k).*conv2(B_out_r(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
        E_out_g(k)= sum(sum(-1*B_out_g(:,:,k).*conv2(B_out_g(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
        E_out_b(k)= sum(sum(-1*B_out_b(:,:,k).*conv2(B_out_b(:,:,k),N_d,'same') + C.*conv2(C,N_d,'same')));
    end
    % final image
    image_out(:,:,1)= image_out_r;
    image_out(:,:,2)= image_out_g;
    
    image_out(:,:,3)= image_out_b;
    % image_out= uint8(image_out);
    imwrite(image_out,'enhanced_image.png');

    % Calculate PSNR
    mse = mean((image_rgb(:) - image_out(:)).^2); % Mean Squared Error
    psnr_values(i) = 10 * log10((255^2) / mse); % PSNR formula: 10 * log10(MAX^2 / MSE)
end

% Plot PSNR vs. alpha
plot(alpha_values, psnr_values, 'bo-');
xlabel('Alpha');
ylabel('PSNR (dB)');
title('PSNR vs. Alpha');
grid on;

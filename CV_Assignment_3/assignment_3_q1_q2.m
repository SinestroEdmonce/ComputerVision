% img1 = './resource/uttower1.jpg';
% img2 = './resource/uttower2.jpg';
% img1 = './resource/image-2frame-region-filled.jpg';
% img2 = './resource/image-4filled.jpg';
% img1 = './resource/image-4mosaic-1.jpg';
% img2 = './resource/image-4mosaic-2.jpg';
img1 = './resource/image-with-noise-4mosaic-1.jpg';
img2 = './resource/image-with-noise-4mosaic-2.jpg';

image_1 = imread(img1);
image_2 = imread(img2);

% [ans1, ans2] = mosaic(image_1,image_2,'manual',4);
% [ans1, ans2] = mosaic(image_1,image_2,'vlfeat');
[ans1, ans2] = mosaic(image_1,image_2,'ransac',30,0.8,10000);

% Show the results
imshow(ans1)
figure;
imshow(ans2)

% func: 'vlfeat', 'ransac', 'manual'
% manual_selected_pts_num: be used as the number of pairs of points to be selected
%              manually
% ransac_threshold: be used to determine the number of points to be
%                    selected and when to exit the loop
function [homography, warpping_image] = mosaic(image_1, image_2, func, manual_selected_pts_num, ransac_threshold, max_loop)
    % Judge the parameter
    if ~exist('manual_selected_pts_num', 'var') || isempty(manual_selected_pts_num)
        manual_selected_pts_num= 4;
    end
    if ~exist('ransac_threshold', 'var') || isempty(ransac_threshold)
        ransac_threshold= 0.1;
    end
    if ~exist('max_loop', 'var') || isempty(max_loop)
        max_loop= 10000;
    end
    if strcmp(func, 'vlfeat') || strcmp(func, 'ransac')
        I1 = single(rgb2gray(image_1)) ;
        I2 = single(rgb2gray(image_2)) ;
        [f1, d1] = vl_sift(I1) ;
        [f2, d2] = vl_sift(I2) ;
        [m, s] = vl_ubcmatch(d1, d2);
        [~, perm] = sort(s, 'ascend') ;
        m = m(:, perm);
        s  = s(perm);
        manual_selected_pts_num = round(size(s,2) * 0.8); 
        x1 = f1(1,m(1,1:manual_selected_pts_num));
        y1 = f1(2,m(1,1:manual_selected_pts_num));
        x2 = f2(1,m(2,1:manual_selected_pts_num));
        y2 = f2(2,m(2,1:manual_selected_pts_num));
        if strcmp(func, 'vlfeat') 
            % Compute the homography matrix parameters
            A = [x1;y1;ones(1,manual_selected_pts_num)];
            B = [x2;y2;ones(1,manual_selected_pts_num)];
            x = B/A;
        elseif strcmp(func, 'ransac') 
            % RANSAC
            best_match_num = -1; 
            for times= 1:max_loop
                randnum = round(rand(1,4)*(manual_selected_pts_num-2))+1;
                A = [x1(randnum);y1(randnum);ones(1,4)];
                B = [x2(randnum);y2(randnum);ones(1,4)];
                x = B/A;
                all_A = [x1;y1;ones(1,manual_selected_pts_num)];
                all_B = x*all_A;
                delta = all_B-[x2;y2;ones(1,manual_selected_pts_num)];
                temp_num = sum(abs(delta(1,:))<10 & abs(delta(2,:))<10);
                if temp_num > best_match_num
                    best_match_num = temp_num;
                end
                if best_match_num > manual_selected_pts_num*ransac_threshold
                    break
                end
            end
            
        end
    elseif strcmp(func, 'manual')
        % Obtain correspondences by recording the mouse clicks
        figure;
        imshow(image_1);
        hold on
        [x1, y1] = ginput(manual_selected_pts_num);
        plot(x1, y1, 'o');

        figure;
        imshow(image_2);
        hold on
        [x2, y2] = ginput(manual_selected_pts_num);
        plot(x2, y2, 'o');
        
        % Compute the homography matrix parameters
        A = [x1';y1';ones(1,manual_selected_pts_num)];
        B = [x2';y2';ones(1,manual_selected_pts_num)];
        x = B/A;
    end

    % Warp between image planes
    [im,dx,dy,im_w,im_h] = apply_warpping_operation(image_1,x);
    homography = uint8(im);

    % Transform the image
    [h_J, w_J, s] = size(image_2);
    new_w = max(w_J, im_w);
    new_h = max(h_J, im_h);
    new_data = -1*ones(new_h,new_w,1);
    new_data = -1*ones(new_h,new_w,2);
    new_data = -1*ones(new_h,new_w,3);
    for idx1 = 1:h_J
        for idx2 = 1:w_J
                new_data(idx1+round(dy),idx2+round(dx),:) = image_2(idx1,idx2,:);
        end
    end
    for idx1 = 1:im_h
        for idx2 = 1:im_w
            if im(idx1,idx2,:) ~= -1
                new_data(idx1,idx2,:) = im(idx1,idx2,:);
            end
        end
    end
    warpping_image = uint8(new_data);
end



% Warpping Function that is used to do the warpping task for images
function [B, dx, dy, B_w, B_h] = apply_warpping_operation(image_1,T)
    [h,w,~] = size(image_1);
    u = [1, w, 1, w;
         1, 1, h, h;
         1, 1, 1, 1];
    v = T*u;
    x_min = round(min(v(1,:)));
    x_max = round(max(v(1,:)));
    y_min = round(min(v(2,:)));
    y_max = round(max(v(2,:)));
    
    if x_min <= 0
        dx = 1-x_min;
    else
        dx = 0;
    end
    if y_min <= 0
        dy = 1-y_min;
    else
        dy = 0;
    end
    B_w = x_max+dx;
    B_h = y_max+dy;
    B = -1*ones(B_h,B_w,1);
    B = -1*ones(B_h,B_w,2);
    B = -1*ones(B_h,B_w,3);

    for idx1 = 1:h
        temp = T*[1:w;idx1*ones(1,w);ones(1,w)];
        temp = round(temp);
        temp(1,:) = temp(1,:) + dx;
        temp(2,:) = temp(2,:) + dy;
        for idx2 = 1:w
            B(temp(2,idx2),temp(1,idx2),:) = image_1(idx1,idx2,:);
        end
    end
    % Fill the blank according to the box 
    for idx1=y_min+dy:y_max+dy
        for idx2=x_min+dx:x_max+dx
            if(B(idx1,idx2,:)==-1)
                
                temp = [idx2-dx;idx1-dy;1];
                temp = T\temp;
                tj = temp(1); 
                ti = temp(2);
                % Fill in the picture only when it among the original area
                if ti>=1 && ti<h && tj >=1 && tj < w
                    v = tj - floor(tj);
                    u = ti - floor(ti);
                    ti = floor(ti);
                    tj = floor(tj);
                    temp(:)=(1-u)*(1-v)*image_1(ti,tj,:)+...
                            u*(1-v)*image_1(ti+1,tj,:)+...
                            (1-u)*v*image_1(ti,tj+1,:)+...
                            u*v*image_1(ti+1,tj+1,:);
                    B(idx1,idx2,:) = temp;
                end
            end
        end
    end
end

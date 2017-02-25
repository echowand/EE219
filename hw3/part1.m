clear all;%load the datadata = importdata('ml-100k/u.data');user_id = data(:, 1);item_id = data(:, 2);row_num = max(user_id);col_num = max(item_id);R = zeros(row_num, col_num);W = zeros(row_num, col_num);for i = 1:size(user_id)    R(data(i, 1), data(i, 2)) = data(i, 3);    W(data(i, 1), data(i, 2)) = 1;end[U1, V1] = wnmfrule(R, 10);error_m1 = W.*(R - (U1*V1)).^2;error1 = sum(error_m1(:));fprintf('k = 10, total least squared error: %f\n', error1);[U2, V2] = wnmfrule(R, 50);error_m2 = W.*(R - (U2*V2)).^2;error2 = sum(error_m2(:));fprintf('k = 50, total least squared error: %f\n', error2);[U3, V3] = wnmfrule(R, 100);error_m3 = W.*(R - (U3*V3)).^2;error3 = sum(error_m3(:));fprintf('k = 100, total least squared error: %f\n', error3);
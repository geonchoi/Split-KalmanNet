clc; clear; close all;

UTIME_Hz = 1e6;

% mode = 'train';
% mode = 'valid';
mode = 'test';

switch mode
    case 'train'
        seq_len = 50;
        gt = load('groundtruth_2012-01-22.csv');
        cov = load('cov_2012-01-22.csv');
        odometry_mu = load("odometry_mu_100hz_2012-01-22.csv");
        x_name = 'train_data_x';
        y_name = 'train_data_y';
    case 'valid'
        seq_len = 200;
        gt = load('groundtruth_2012-01-22.csv');
        cov = load('cov_2012-01-22.csv');
        odometry_mu = load("odometry_mu_100hz_2012-01-22.csv");
        x_name = 'valid_data_x';
        y_name = 'valid_data_y';
    case 'test'
        seq_len = 2000;
        gt = load('groundtruth_2012-04-29.csv');
        cov = load('cov_2012-04-29.csv');
        odometry_mu = load("odometry_mu_100hz_2012-04-29.csv");
        x_name = 'test_data_x';
        y_name = 'test_data_y';
end


get_gt = @(v) interp1(gt(2:end, 1), gt(2:end, 2:end), v, 'nearest');
get_odom = @(v) interp1(odometry_mu(2:end, 1), odometry_mu(2:end, 2:end), v, 'nearest');

t_cov = cov(2:end, 1);


t_init = t_cov(1);
t_end = t_cov(end);
t_sample = (t_init:UTIME_Hz/1:t_end).';
t_sec = (t_sample - t_sample(1)) / (UTIME_Hz/1);
dt = t_sec(2:end) - t_sec(1:end-1);

pose_gt = get_gt(t_sample);
x = pose_gt(:, 1);
y = pose_gt(:, 2);
vx = zeros(length(dt)+1, 1);
vy = zeros(length(dt)+1, 1);
vx(2:end) = (x(2:end) - x(1:end-1)) ./ dt;
vy(2:end) = (y(2:end) - y(1:end-1)) ./ dt;
accx = zeros(length(dt)+1, 1);
accy = zeros(length(dt)+1, 1);
accx(2:end-1) = (vx(3:end) - vx(2:end-1)) ./ dt(2:end);
accy(2:end-1) = (vy(3:end) - vy(2:end-1)) ./ dt(2:end);

odom_data = get_odom(t_sample);
odom_vx = zeros(length(dt)+1, 1);
odom_vy = zeros(length(dt)+1, 1);
odom_vx(2:end) = (odom_data(2:end, 1) - odom_data(1:end-1, 1)) ./ dt;
odom_vy(2:end) = (odom_data(2:end, 2) - odom_data(1:end-1, 2)) ./ dt;


% data_x_tmp = zeros(4, length(t_sample));
% data_x_tmp(1,:) = x.';
% data_x_tmp(2,:) = y.';
% data_x_tmp(3,:) = vx.';
% data_x_tmp(4,:) = vy.';
data_x_tmp = zeros(6, length(t_sample));
data_x_tmp(1,:) = x.';
data_x_tmp(2,:) = y.';
data_x_tmp(3,:) = vx.';
data_x_tmp(4,:) = vy.';
data_x_tmp(5,:) = accx.';
data_x_tmp(6,:) = accy.';

data_y_tmp = zeros(2, length(t_sample));
data_y_tmp(1,:) = odom_vx.';
data_y_tmp(2,:) = odom_vy.';



total_len = length(t_sample);
test_data_num = floor(total_len / seq_len);

% data_x = zeros(test_data_num, 4, test_seq_len);
data_x = zeros(test_data_num, 6, seq_len);
data_y = zeros(test_data_num, 2, seq_len);
for i = 1:test_data_num
    data_x(i,:,:) = data_x_tmp(:, 1:seq_len);
    data_y(i,:,:) = data_y_tmp(:, 1:seq_len);
    data_x_tmp = data_x_tmp(:, seq_len+1:end);
    data_y_tmp = data_y_tmp(:, seq_len+1:end);
end

save(x_name, 'data_x')
save(y_name, 'data_y')


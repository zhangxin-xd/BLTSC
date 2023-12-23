function [ output_srm ] = nonlinear( input_srm ,lamda_1 ,max )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    [row,col]=size(input_srm);
    N = row*col;
    lamda = lamda_1;
    Weight = ones(1,N);
    y_old = ones(1,N);
    max_it = max;
    epsilon = 1e-6;
    im_src=reshape(input_srm,1,N);
    for T=1:1:max_it
        im_src=im_src.*Weight;
        Weight = 1 - 2.71828.^(-lamda*im_src);
        Weight(Weight<0) = 0;
     
        res = norm(y_old)^2/N - norm(im_src)^2/N;
        y_old = im_src;
     
        if (abs(res)<epsilon)
            break;
        end
    end
    output_srm=reshape(im_src,row,col);

end


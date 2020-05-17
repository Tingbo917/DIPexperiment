
clc;
clear;
%读入图像，并转换为double型
I_1=imread('C:\Users\Administrator\Desktop\img\danghui_yuantu.BMP');
I_2=imread('C:\Users\Administrator\Desktop\img\dayanta_yuantu.BMP');
I_3=imread('C:\Users\Administrator\Desktop\img\xust_yuantu.BMP');

I_D_1=im2double(I_1);
I_D_2=im2double(I_2);
I_D_3=im2double(I_3);

[M_1,N_1]=size(I_D_1);
[M_2,N_2]=size(I_D_2);
[M_3,N_3]=size(I_D_3);

a=0;
b=0.10;
B=1;

N_Ray1_1=a+b*raylrnd(B,M_1,N_1);
J_rayl_1=I_D_1+N_Ray1_1;

N_Ray1_2=a+b*raylrnd(B,M_2,N_2);
J_rayl_2=I_D_2+N_Ray1_2;

N_Ray1_3=a+b*raylrnd(B,M_3,N_3);
J_rayl_3=I_D_3+N_Ray1_3;

imwrite(J_rayl_1,'C:\Users\Administrator\Desktop\img\ruili\danghui_ruili.BMP');
imwrite(J_rayl_2,'C:\Users\Administrator\Desktop\img\ruili\dayanta_ruili.BMP');
imwrite(J_rayl_3,'C:\Users\Administrator\Desktop\img\ruili\xust_ruili.BMP');

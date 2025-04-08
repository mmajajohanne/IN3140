clc; clear; close all;
syms c1 s1 c2 s2 c3 s3 a2 a3 c23 s23

J_v = [-a2*c2*s1-a3*s1*c23, -c1*(a2*s2+a3*s23), -c1*a3*s23;
    a2*c1*c2+a3*c1*c23, -s1*(a2*s2+a3*s23), -s1*a3*s23;
    0,a2*c2+a3*c23,a3*c23];
       

% Compute determinant of J_v
det_Jv = det(J_v);


% Display result
disp('Determinant of J_v:');
disp(simplify(det_Jv));
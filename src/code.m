clc;
clear all;
 
nx=20;
ny=100;
LbyD=1;
 
theta=linspace(0,2*pi,ny+1);
z=linspace(0,1,nx);
dtheta = theta(2)-theta(1);
dz = z(2)-z(1);
 
 
eps=[0.2 0.4 0.6 0.8];
for k=1:4
    h = zeros(1,ny);
    for hh = 1:ny
        h(hh) = 1+((eps(k))*cos(theta(hh)));
    end
 
    P=zeros(nx,ny);
    error=1;
    while error>1e-5
    p_0=P;
        for i=2:nx-1
            for j=2:ny-1;
            P(i,j)=(((3/h(j))*(P(i,j+1)-P(i,j-1))*(1/(2*dtheta))^2*(h(j+1)-h(j-1)))-((1/h(j))^3*(h(j+1)-h(j-1))*(1/(2*dtheta)))+(((P(i,j+1)+P(i,j-1))/(dtheta^2))+((1/LbyD^2)*(P(i+1,j)+P(i-1,j))*(1/dz^2))))/((2/dtheta^2)+(1/LbyD^2)*(2/dz^2));
            end
            for j=1
                P(i,1)=((3/h(j))*(P(i,j+1)-P(i,100))*(1/(2*dtheta))^2*(h(j+1)-h(100))-(1/h(j))^3*(h(j+1)-h(100))*(1/(2*dtheta))+((P(i,j+1)+P(i,100))/(dtheta^2))+((1/LbyD^2)*(P(i+1,j)+P(i-1,j))*(1/dz^2)))/((2/dtheta^2)+(1/LbyD^2)*(2/dz^2));
            end
            for j=ny
                P(i,ny)=((3/h(j))*(P(i,1)-P(i,j-1))*(1/(2*dtheta))^2*(h(1)-h(j-1))-(1/h(j))^3*(h(1)-h(j-1))*(1/(2*dtheta))+((P(i,1)+P(i,j-1))/(dtheta^2))+((1/LbyD^2)*(P(i+1,j)+P(i-1,j))*(1/dz^2)))/((2/dtheta^2)+(1/LbyD^2)*(2/dz^2));
            end
% Modified by Michael Reiter, 22.7.05
clear all;
global ID  Beta Rho n G Dep CurlyEpsilon permval permprob tranval tranprob  R W AlphaVec
ID='macro';
Beta=0.96;   % time preference
Rho=2;       % coefficient of relative risk aversion
n=20;        % number of grid points
G=1.01;      % permanent income growth rate
Dep=0.90;    % depreciation rate
CurlyEpsilon=0.36;  % Cobb-Douglas production function parameter
permval=[0.90,1.00,1.10];   % permanent shock values
permprob=[0.25,0.50,0.25];  % permanent shock probabilities
tranval=[1.00];             % transitory  shock values
tranprob=[1.00];            % transitoty shock probabilities
R=inline('1+CurlyEpsilon.*k.^(CurlyEpsilon-1)','k','CurlyEpsilon');   % set up interest rate function
W=inline('(1-CurlyEpsilon).*(k.^CurlyEpsilon)','k','CurlyEpsilon');   % set up wage rate function
kSS=((((G^Rho)/(Beta*Dep))-1)/CurlyEpsilon)^(1/(CurlyEpsilon-1));
aSS=kSS*G/Dep;
AlphaVec=exp(linspace(0.00,log(3*aSS),n))-1;       % set up grid points

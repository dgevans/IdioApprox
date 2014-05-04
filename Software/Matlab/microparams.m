% Modified by Michael Reiter, 22.7.05
clear all
global ID Rho Beta n G p R W permval permprob tranval tranprob  AlphaVec Dep % CapGamma parameters that are global
ID='Micro';
Beta=0.96;   % time preference
Rho=2;       % coefficient of relative risk aversion
n=20;        % number of grid points
G=1.03;      % permanent income growth rate
p=0.005;     % probability of zero-income; set this to zero for liquidity constrained soln
if p==0.0
ID='LiqCo';  % if zero-income probability is zero, liquidity constraint may be binding.
end;

permval=[0.90,1.00,1.10];   % permanent shock values
permprob=[0.25,0.50,0.25];  % permanent shock probabilities
if ID=='LiqCo' 
tranval=permval;     % transitory  shock values
tranprob=permprob;   % transitory shock probabilities
else
tranval=[permval/(1-p),0.0];   % transitory  shock values
tranprob=[permprob*(1-p),p];   % transitoty shock probabilities
end;
R=inline('1.04','x','y');     %  set up interest rate function, which is a constant in micro version 
W=inline('1.00','x','y');     %  set up wage rate function , which is a constant in micro version
AlphaVec=exp(exp(exp(linspace(0.00,log(log(log(10+1)+1)+1),n))-1)-1)-1;  % set up triple exponential grid

Dep=1;

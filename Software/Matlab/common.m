% Modified by Michael Reiter, 22.7.05
% two other user-defined M-file functions, GothVP and Cnext, are also involved in running this
% program.  
global ID Rho Beta n AlphaVec NP UP C M  % parameters that are global
NP=inline('c.^(-1/Rho)','c','Rho');
UP=inline('c.^(-Rho)','c','Rho');
PeriodsToAdd=99;         % number of extra periods to add onto initial C=Y solution
C=(0:n-1)'; 
M=(0:n-1)'; 
for l=1:PeriodsToAdd
  % calculate ct from each grid point in AlphaVec
  ChiVec=NP(GothVP(AlphaVec),Rho); % inverse Euler equation
  MuVec =AlphaVec+ChiVec;
  M=[M, MuVec'];             % matrix of interpolation data
  C=[C,ChiVec'];             % matrix of interpolation data
end;
if ID=='LiqCo'
  zero=(linspace(0,0,PeriodsToAdd+1));  % produce a vector that is made up of zeroes
  C=[zero',C']';  % zero is added to each period's interpolation data to handle liquidity constraints
  M=[zero',M']';  % zero is added to each period's interpolation data to handle liquidity constraints
end;
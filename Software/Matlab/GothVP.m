% Modified by Michael Reiter, 22.7.05
function EUP=GothVP(a)
% GothVP is defined as Gothic V prime in the paper (end of period marginal value). 
% Another M-file function, Cnextp, is also involved in running this program
global ID Beta Rho CurlyEpsilon Dep G R W UP NP tranval tranprob permval permprob 

% EUP is used to take the sum of marginal utilities UP weighted by probabilities
EUP=zeros(size(a)); 

for i=1:length(tranval)
  for j=1:length(permval)
    CapPsi=permval(j);
    CapXi=tranval(i);
    k=a.*Dep./(G.*CapPsi);
    EUP=EUP+Dep*Beta.*R(k,CurlyEpsilon) ... 
	.*UP(G.*CapPsi.*Cnextp(R(k,CurlyEpsilon).*k+W(k,CurlyEpsilon).*CapXi),Rho).*tranprob(i)...
	.*permprob(j);
  end;
end;

if ID=='macro' 
  izero = a==0.0;
  EUP(izero) = Inf;
end;


% This program graphs the consumption function for unemployment model

clear all
global M C MuVec ChiVec
microparams;
common;
plot(M(1:5,1),C(1:5,1),MuVec,ChiVec)
text(4,1.2,'Unemp Model')
text(2,1.6,'45 Degree')
xlabel('m')
ylabel('c')
title('Consumption Function for Micro Unemp Model')

param n:= 10000; #Cantidad de puntos
param u1:= 1000; #rango inferior para el w
param u2:= 5000; #rango superior para el w

#####################   Conjunto(s)   ###########################

set I:= 1..n;

#####################   Parámetros    ###########################
param w{i in I} = Uniform(u1, u2);  
param x{i in I} = round(Uniform(0, n));
param y{i in I} = round(Uniform(0, n)); 

#####################   Variables   #############################
var Xg>= 0;
var Yg>= 0;

#########################   FO  ################################

minimize FO: sum{i in I}(w[i]*sqrt((x[i]-Xg)**2 + (y[i]-Yg)**2));

########   Restricciones: ninguna, problema irrestricto   #######


model TransientConduction
import SI = Modelica.Units.SI;

// Material Properties
parameter SI.Density rho = 7850  "material density [kg/m3]";
parameter SI.SpecificHeatCapacity c = 0.47e3 "material heat capacity [J/(KgK)]";
parameter SI.CoefficientOfHeatTransfer h = 300 "coefficient of convective heat transfer [W/(m2 K)]";
parameter SI.ThermalConductivity k = 12.1e-2 "thermal conductivity of material [W/(m K)]";

// Geometry
parameter SI.Length r0 (displayUnit = "mm") = 1e-2 "sphere radius";

// Thermodynamic Parameters
final parameter SI.ThermalDiffusivity alpha = k/(rho*c) "thermal diffusivity";
final parameter SI.Length L = r0/3 "Heat conduction characteristic length"; 
final parameter Real Bi = h*L/k "Biot number";
final parameter SI.Frequency Fo = alpha/L^2 "Fourier Number";
 
// Boundary Conditions
parameter SI.Temperature T_init = 300 "Sphere initial uniform temperature";
parameter SI.Temperature T_inf = 353 "Heating/Cooling fluid bulk temperature"; 

// Lumped parameters formulation
SI.Temperature T_lumped "Sphere temperature - lumped parameters";
Real theta_lumped "adimensional sphere temperature - lumped parameters";

// Finite Volume formulation
parameter Integer N = 20 "number of volumes";
SI.Temperature T[N] "Spherical shells temperatures - finite volume"; 
Real theta[N] "adimensional temperatures";
SI.Length r[N] "spherical shells radiii";
Real x[N] "adimensional spherical shells radii";
Real ql[N]"adimensional heat transfer from previous shell";
Real qr[N]"adimensional heat transfer to next shell";

initial equation
// set sphere initial temperature - uniform
T_lumped = T_init;
T = T_init*ones(N);

equation

// adimentional formulation
theta_lumped = (T_lumped - T_inf)/(T_init - T_inf);

for j in 1:N loop
  theta[j] = (T[j] - T_inf)/(T_init - T_inf);
  x[j] = j/N;
  r[j] = x[j]*r0;
end for;

// lumped parameters energy balance equation
der(theta_lumped) = -Bi*Fo*theta_lumped;



// finite volumes energy balance equations
for j in 1:N loop
  ql[j] = if j ==1 then 0 else -x[j-1]^2/(x[j]^3 - x[j-1]^3)*(theta[j]-theta[j-1])/(x[j]-x[j-1]);
  qr[j] = x[j]^2/(x[j]^3 - (if j ==1 then 0 else x[j-1]^3))*(if j == N then -3*Bi*theta[N] else (theta[j+1]-theta[j])/(x[j+1]-x[j]));
  der(theta[j]) = Fo/3*(ql[j] + qr[j]);
end for;


annotation(
    experiment(StartTime = 0, StopTime = 3000, Tolerance = 1e-06, Interval = 6));
end TransientConduction;

Gc_pll=-K_Factor(-Gp_pll,BW_pll,PM_pll);

Z_pll=zero(Gc_pll);     % Controller zero
res = pole(Gc_pll);
P_pll=res(1);           % Controller pole
K_pll=dcgain(Gc_pll/tf([1 -Z_pll],[1 -P_pll 0])); % Controller DC gain
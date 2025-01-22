# -*- coding: utf-8 -*-
"""
Created on Tue Oct 9 10:59 2024

@author: schwarz
"""

import numpy as np
import scipy
from scipy.optimize import minimize
from TILMedia import VLEFluid
from icecream import ic

R = scipy.constants.R

class Adsorbent:
    '''
    This class models the equilibrium behaviour of a sorbent with a working fluid.
    Per default silicagel123 and water are used and the equilibrium is modeled
    via Dubinin.
    
    Furthermore this class is used to calculate initial temperatures for an
    LTJ experiment. Therefore isosteres can be drawn with this class.
    '''
    
    def __init__(self, vlefluid = VLEFluid("Water", computeVLEAdditionalProperties=True)):
        self.ads = vlefluid # Adsorbate, per default water
        
        self.M = self.ads.M*1000 # in g/mol
        #Anfangswerte der Parameter für die Charakteristische Kurve für 
        # [w] = cm^3/g und [A] = J/g
        #self.c0 = [5.072313e-1, 1.305531e+2, -8.492403e+1, 4.128962e-3] # Schawe
        
        self.T = 273.15 # K, temperature of the adsorbent
        self.p = 100_000 # Pa
        
        self.ads.setState_pTxi(self.p, self.T)
        
        self.c_p_dry = 1 # J/gK from Schawe01
        self.c_p_ads = 0 # J/gK
        self.m = 1 # g
        self.x = 0 # g/g
        self.W = 0 # cm^3/g
        self.c_p = 1 # J/gK
        
        # Fitted parameters for Dubinin EQ-model
        #          cm^3/g             J/g                 J/g                 cm^3/g
        self.c0 = [0.477306939792391, 133.06286794344513, -93.03831417288345, -0.006761042963097699] 
        
        # Values to calculate the adsorption cycle
        self.p_ads = 0 # Pa
        self.p_des = 0
        self.p_sat_T_des = 0
        
        self.T_vap = 0
        self.T_ads_end = 0
        self.T_des_end = 0
        
        self.x_ic = 0 # g/g
        self.x_ih = 0
        
        self.A_ads_end = 0 # J/g
        self.A_des_end = 0
        
        self.W_ads_end = 0 # cm^3/g
        self.W_des_end = 0
        
    def get_c_p_dry(self):
        return self.c_p_dry
    
    def set_c_p_dry(self, c_p):
        if c_p <= 0:
            print(f'Values for c_p must be greater than 0. c_p: {c_p}')
            return
        
        self.c_p_dry = c_p
    
    def get_c_p(self):
        # TODO Model for heat capacity is not complete
        return self.x*self.ads.VLE.cp_l / 1000 + (1-self.x)*self.c_p_dry # J/gK
    
    
    def get_c_p_ads(self):
        
        self.ads.setState_pTxi(self.p, self.T)
        c_p_v = self.ads.VLE.cp_v
        
        def h_ads(T, p): return self.adsorptionPotential(p, T)
        
        dh_ads = self.d(h_ads, self.T, args=[self.p])
        dh_ads_dT = dh_ads[0] / dh_ads[1]
        
        return c_p_v - dh_ads_dT
        
        
    def get_m(self):
        return self.m
    
    def set_m(self, m):
        if m <= 0:
            print(f'Values for m must me greater than 0. m: {m}')
            return
        
        self.m = m

    def adsorptionPotential(self, p_ad, T):
        p_sat = self.get_p_sat(T)
        return R/self.M * T * np.log(p_sat/p_ad) # J/g


    def adsorptionEnthalpy(self, p_ad, T):
        A = self.adsorptionPotential(p_ad, T)
        dh_v = self.evapEnthalpy(p_ad, T)
        return -dh_v - A # J/g
    
    
    def evapEnthalpy(self, p, T):
        self.ads.setState_pTxi(p, T)
        return (self.ads.VLE.h_v - self.ads.VLE.h_v) / 1000 # J/g

    def T_from_adsPot(self, p_sat, p_ad, A):
        return A* self.M/R * 1/(np.log(p_sat/p_ad)) # K

    def p_ads_from_adsPot(self, T, A):
        p_sat = self.get_p_sat(T)
        return p_sat* 1/(np.exp(A * self.M/R * 1/T)) # Pa
    
    def get_T(self, p, load):
        '''
        This function returns the equilibrium temperature corresponding to a
        pressure (Pa) and loading (g/g)
        '''
        
        return minimize(self.min_x, x0=323.15,\
                        bounds=[(200, 373.15)],\
                        args=(p, load)).x[0]
    
    # Characteristic curve by Dubinin
    # [A] = J/g
    def dubinin(self, A, a, b, c, d):
        return d+a/np.pi*(np.arctan((A-b)/c)+np.pi/2) # cm^3/g
    
    def get_x(self, T, p): # K, Pa
        return self.get_rho(T, p)*self.dubinin(self.adsorptionPotential(p, T), *self.c0)*1000 # g/g
    
    def min_x(self, T, p, x):
        '''This is the cost function to calculate T only from p and x
        T: Temperature in K
        p: Pressure in Pa
        x: absolute loading in g/g
        '''
        
        return abs(x-self.get_x(T, p))
        
    def reverse_dubinin(self, W, a, b, c, d):
        '''Returns the adsorption potential for a given specific adsorbed volume
        W: specific adsorbed volume in cm^3/g
        '''
        
        # [W] = cm^3/g
        return c*(np.tan((np.pi/a) * (W-d) - (np.pi/2)))+b # J/g

    def get_rho(self, T, p):
        self.ads.setState_pTxi(p, T)
        return (self.ads.VLE.d_l) / 1_000_000 # kg/cm^3

    def get_p_sat(self, T):
        self.ads.setState_Txi(T)
        return self.ads.VLE.p_v # Pa

    def set_temperatures(self, T_vap, T_ads, T_des):
        # Input in °C
        self.T_vap = T_vap + 273.15 # K
        self.T_ads_end = T_ads + 273.15 # K
        self.T_des_end = T_des + 273.15 # K

    def set_pressures(self):
        p_sat = []
        for T in [self.T_vap, self.T_ads_end, self.T_des_end]:
            p_sat.append(self.get_p_sat(T)) # in Pa

        self.p_ads = p_sat[0]
        self.p_des = p_sat[1]
        self.p_sat_T_des = p_sat[2]

        ic("Pressures set")
        ic(self.p_ads, self.p_des, self.p_sat_T_des)

    def set_adsPot(self):
        self.A_ads_end = self.adsorptionPotential(self.p_ads,\
                                                  self.T_ads_end)
        self.A_des_end = self.adsorptionPotential(self.p_des,\
                                                  self.T_des_end)
                                             
        ic("Adsorption potential set")
        ic(self.A_des_end, self.A_ads_end)

    def set_W(self):
        self.W_ads_end = self.dubinin(self.A_ads_end, *self.c0)
        self.W_des_end = self.dubinin(self.A_des_end, *self.c0)

        ic("Loading W set")
        ic(self.W_ads_end, self.W_des_end)

    def set_x_load(self):
        self.x_ih = self.get_rho(self.T_ads_end, self.p_ads)*self.W_ads_end*1000 # g/g
        self.x_ic = self.get_rho(self.T_des_end, self.p_des)*self.W_des_end*1000

        ic("Loading x set")
        ic(self.x_ih, self.x_ic)
        
    def cal_T_lumped_t(self, alpha, A, t, T_carrier, T_0):
        '''
        Calculate the temperature of the sorbent (assuming lumped parameter model)
        relative to alpha, Area A, time, temperature of the carrier and starting temp
        '''
        return T_carrier - (T_carrier- T_0)*np.exp(-(alpha*A)/(self.m*self.c_p)*t)
    
    def cost_of_T_lumped(self, x, A, t, T_carrier, T_0, T_sam):
        
        
        return T_sam - self.cal_T_lumped_t(x, A, t, T_carrier, T_0)
    
    def get_alpha(self, A, t, T_carrier, T_0, T_sam, alpha_0):
        
        alpha = minimize(self.cost_of_T_lumped, alpha_0, args=(A, t, T_carrier,\
                                                              T_0, T_sam))
            
        return alpha.x[0]

    def calibrate_alpha(self, A, t, T_carrier, T_sam, alpha_0):
        T_0 = T_sam[0]
        
        alpha = []
        for i in range(len(T_carrier)):
            alpha.append(self.get_alpha(A, t[i], T_carrier[i], T_0, T_sam[i], alpha_0))
            
        return np.mean(alpha), max(alpha), min(alpha)

    def get_isostere(self, x_load, p_start, p_end, T_start, dT = 1, verbose=False):
        '''
        Creates an isostere between p_ads and p_des from T_start
        At the end of the isostere T_initial is reached
        Output consists of T_initial and all calculated points of the isostere
        '''
        ic.enable()
        
        ic("Creating isostere")
        ic(x_load, p_start, p_end, T_start, dT)
        
        if not verbose: ic.disable()
        greater_p = False
        
        x_load = x_load / 1000 # kg/g

        p = p_start
        T = T_start
        
        p_list = []
        T_list = []
        
        if p_start < p_end:
            greater_p = False
            dT = dT
        elif p_start > p_end:
            greater_p = True
            dT = -dT
        else:
            print(f"p_start: {p_start} and p_end {p_end} have the same value")
            print("Aborting program")
            return
        
        dp = abs(p_start-p_end)
        
        while (greater_p and p >= p_end or not greater_p and p <= p_end):
            T_list.append(T)
            p_list.append(p)
            
            rho = self.get_rho(T, p)
            W = x_load / rho
            A = self.reverse_dubinin(W, *self.c0)
            p = self.p_ads_from_adsPot(T, A)
            ic(p, A, W, x_load, rho)
            
            T = T + dT
            
        return T, (T_list, p_list), x_load*1000

    def get_initial_temperatures(self, T_eva, T_ads, T_des, return_isosteres=False):
        '''Get initial temperatures for an LTJ experiment
        This function returns the initial temperatures for adsorption and
        desorption for a given temperature set. Furtherore the two isosteres
        that were used to calculate these temperatures are returned as well.
        '''

        self.set_temperatures(T_eva, T_ads, T_des)
        self.set_pressures()
        self.set_adsPot()
        self.set_W()
        self.set_x_load()

        isostere_ic = self.get_isostere(self.x_ic,\
                                        self.p_des,\
                                        self.p_ads,\
                                        self.T_des_end,\
                                        dT = 0.01)
        
        isostere_ih = self.get_isostere(self.x_ih,\
                                        self.p_ads,\
                                        self.p_des,\
                                        self.T_ads_end,\
                                        dT = 0.01)

        print("Initial temperatures for Adsorption and Desorption")
        print(f"T_ads_init: {isostere_ic[0]-273.15}\nT_des_init: {isostere_ih[0]-273.15}")
        
        if not return_isosteres:
            T_ads_init = isostere_ic[0]
            T_des_init = isostere_ih[0]
            return T_ads_init, T_des_init
        
        return isostere_ic, isostere_ih
    
    
    def d(self, f, x, args, dx_r=0.01):
        '''
        Calculates the differential of f at x for a deviation
        of x of dx_r (by default 1 %)
        '''
        
        dx = dx_r * x
        df = f(x + dx/2, *args) - f(x - dx/2, *args)
        
        return df, dx


##############################################################################
######### TEST FUNCTIONS #####################################################
##############################################################################


def test():
    print("Running test with temperature set:")
    print("15, 27, 75")
    ic.disable()
    sor = Sorbens()
    print(sor.get_initial_temperatures(15, 27, 75))
    
def isostere_test():
    print("Running test to create isostere")
    print("Start at p: 100 Pa, x: 0,1")
    
    sor = Sorbens()
    isostere = sor.get_isostere(0.1/1000, 100, 10000, sor.get_T(100, 0.1), dT=0.01)
    
def c_p_ads_test():
    print("Running test to calculate c_p_ads")
    print("For 20 °C and 5000 Pa")
    
    sor = Sorbens()
    sor.T = 273.15 + 20
    sor.p = 5000
    
    print(sor.get_c_p_ads())

if __name__ == "__main__":
    c_p_ads_test()

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 01:22:28 2023

@author: carlo
"""

import re

cities = {'CALOOCAN' : 'CALOOCAN', 
          'LAS PIÑAS' : 'LAS PIÑAS', 
          'LAS PINAS' : 'LAS PIÑAS', 
          'MAKATI' : 'MAKATI', 
          'MALABON' : 'MALABON', 
          'MANDALUYONG' : 'MANDALUYONG', 
          'MANILA' : 'MANILA', 
          'MARIKINA' : 'MARIKINA', 
          'MUNTINLUPA' : 'MUNTINLUPA', 
          'NAVOTAS' : 'NAVOTAS', 
          'PARAÑAQUE' : 'PARAÑAQUE', 
          'PARANAQUE' : 'PARAÑAQUE',
          'PASAY' : 'PASAY', 
          'PASIG' : 'PASIG', 
          'PATEROS' : 'PATEROS', 
          'Q(UEZON)?.*C(ITY)?' : 'QUEZON',
          'SAN JUAN' : 'SAN JUAN', 
          'TAGUIG' : 'TAGUIG', 
          'VALENZUELA' : 'VALENZUELA'}

provinces = {'RIZAL' : 'RIZAL', 
             'LAGUNA' : 'LAGUNA', 
             'BULACAN' : 'BULACAN',
             'CAVITE' : 'CAVITE' , 
             'TAYTAY' : 'RIZAL', 
             'TARLAC' : 'TARLAC',
             'NUEVA ECIJA' : 'NUEVA ECIJA', 
             'PAMPANGA' : 'PAMPANGA',
             'BATANGAS' : 'BATANGAS', 
             'QUEZON' : 'QUEZON', 
             'BATAAN' : 'BATAAN',
             'AURORA' : 'AURORA', 
             'ZAMBALES' : 'ZAMBALES', 
             'GENERAL TRIAS' : 'CAVITE',
             'CARMONA': 'CAVITE'}

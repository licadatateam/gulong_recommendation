# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:37:21 2023

@author: carlo
"""

import re
import numpy as np

color_corrections = {'BIEGE' : 'BEIGE',
                     '(NA|OTHER(S)?|NONE|VARIOUS)' : '',
                     'SILVER' : 'SILVER',
                     'WHITE' : 'WHITE',
                     'BLU(E)?' : 'BLUE',
                     'BLA(C)?(K)?' : 'BLACK',
                     'GR[EA][YT]' : 'GRAY',
                     'RED' : 'RED',
                     'BLUE': 'BLUE',
                     'BROWN' : 'BROWN',
                     'GR(E+)N' : 'GREEN',
                     'ORANGE': 'ORANGE',
                     'MA(R+)(O+)N' : 'MAROON',
                     'YE(L+)OW' : 'YELLOW',
                     'PINK' : 'PINK',
                     'VIOLET' : 'VIOLET'
                     }
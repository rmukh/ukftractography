#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:29:54 2019

@author: rinat
"""

import pandas as pd

l1 = pd.read_csv('log.txt', skiprows=37, skipfooter=117, header=None, sep='\s+', engine='python')

#omit directions colums
l1 = l1.drop(columns=[0,1,2,7,8,9,14,15,16,21,22,23,24])

lessthan = l1[l1 >= 2000].count()
lessthan_percent = lessthan.apply(lambda x: 100 * x / l1.shape[0])
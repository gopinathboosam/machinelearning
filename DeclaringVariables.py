# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:13:07 2020

@author: BoosamG
"""

#Declaring the vairables
a = 100
print(a)

#re-declaring 
a = "adsasdad"
print(a)


a = "Raj"
b = 14
print(a+str(b))


#Global vs Local variables

a = 140
print(a)


def changeA():
    #a = "Changed A"
   
    global a
    print(a)
    a = "Changing Global"
    
changeA()
print(a)

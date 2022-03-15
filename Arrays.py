# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:47:29 2020

@author: BoosamG
"""

#Python Arrays
import array as myarray
a = myarray.array('i',[2,3,4])

b = myarray.array('d',[2.3,3.4,33.3,4.6])

#Accessing the elements
print(b[2])
print(b[-3])

#by using the ':' operator.
print(b[1:3])

#Inserting elements
b.insert(5,3.2);
print(b)

b.insert(2,4.0)
print(b)

#Modify Elements
b[1]= 4.44
print(b)

#Pop element
b.pop(4)
print(b)

#remove
b.remove(2.3)
print(b)

b.reverse()
print(b)


#search by value and get index
print (b.index(4.0))

print(b)
b.insert(3,3.2)
print(b)

print(b.count(3.2))
#Traverse ab array
for k in b:
    print(k)
    

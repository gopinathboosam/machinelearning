# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:42:54 2020

@author: BoosamG
"""

##Tuple
a = (10,20,30);
b = (10,);
print(a,b)
c = (); #Empty Tuple

tup1 = ("Rajesh","2000",1)
print (tup1[2])
tup2 = (1,2,3,4,5,6)
print(tup2[2:4])


#Tuple Packing and Unpacking
y = ("Gopi", 29, "Ags")#Packing
print(y)

(name,age,company) = y
print(y)
print(name)#Unpacking

#Dictionary
a = {'x':100, 'y':200}
b = a.items()
print (b) 

#To perform different task, tuple allows you to use many built-in functions like all(), any(), enumerate(), max(), min(), sorted(), len(), tuple(), etc

#Dictionary'
dict = {"Rajesh":23, "Ramesh":32}
print(dict["Rajesh"])

#Copy 
dictCopy = dict.copy()
print(dictCopy)

#Update
dict.update({"Suresh":33})
print(dict)

#delete
del dict["Ramesh"]
print(dict)
#Listing the dictionary items
print(dict.items())
print(list(dict.items()))

#Cheking whether the key exists
Dict = {'Tim': 18,'Charlie':12,'Tiffany':22,'Robert':25}
Boys = {'Tim': 18,'Charlie':12,'Robert':25}
Girls = {'Tiffany':22}
for key in Dict.keys():
    if key in Boys.keys():
        print("Available")
    else:
        print("Unavailable")

#Sorting dictionary
Students = list(Dict.keys())
print(Students)
Students.sort()
print(Students)

for s in Students:
    print(":".join((s,str(Dict[s]))))

#Other operations and methods of dictionary
#Restrictions on Key Dictionaries
#How to append an element to a key in a dictionary with Python?
my_dict = {"Name":[],"Address":[],"Age":[]};
my_dict["Name"].append("Guru")
my_dict["Address"].append("Mumbai")
my_dict["Age"].append(30)	
print(my_dict)

#Accessing elements of a dictionary
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}
print("username :", my_dict['username'])
print("email : ", my_dict["email"])
print("location : ", my_dict["location"])
#Deleting element(s) in a dictionary

my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}
del my_dict['username']  # it will remove "username": "XYZ" from my_dict
print(my_dict)
my_dict.clear()  # till will make the dictionarymy_dictempty
print(my_dict)
del my_dict # this will delete the dictionarymy_dict
print(my_dict)
#Deleting Element(s) from dictionary using pop() method
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}
my_dict.pop("username")
print(my_dict)
#Appending element(s) to a dictionary
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}

my_dict['name']='Nick'

print(my_dict)
#Updating existing element(s) in a dictionary
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}

my_dict["username"] = "ABC"

print(my_dict)
#Insert a dictionary into another dictionary
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Washington"}

my_dict1 = {"firstName" : "Nick", "lastName": "Price"}

my_dict["name"] = my_dict1

print(my_dict)



























tup1 = ('Robert', 'Carlos','1965','Terminator 1995', 'Actor','Florida');
tup2 = (1,2,3,4,5,6,7);
print(tup1[0])
print(tup2[1:4])


x = ("Guru99", [23,23] , 20, "Education")    # tuple packing
(company,dummy, emp, profile) = x    # tuple unpacking
print(company)
print(emp)
print(profile)
print(dummy)


Dict = {'Tim': 18,'Charlie':12,'Tiffany':22,'Robert':25}   
print((Dict['Tiffany']))

Dict = {'Tim': 18,'Charlie':12,'Tiffany':22,'Robert':25}	
Dict.update({"Sarah":9})
print(Dict)


Dict = {'Tim': 18,'Charlie':12,'Tiffany':22,'Robert':25}	
print("Students Name: %s" % list(Dict.items()))


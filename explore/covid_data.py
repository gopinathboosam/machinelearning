# -*- coding: utf-8 -*-

import requests

##list of Countries:
payload={}
headers = {}
countiresAPI = "https://api.covid19api.com/countries"
listofCountries = requests.request("GET", countiresAPI, headers=headers, data=payload).json()
print(listofCountries)
for countryObj in listofCountries:
    print(countryObj["Country"])
    print(countryObj["ISO2"])


#Get summary
a = "https://api.covid19api.com/summary"
payload={}
headers = {}
summary = requests.request("GET", a, headers=headers, data=payload).json()
print(summary)

##Get listofCountries as Primary
##Get Summary as Secondary
#Iterate through the for loop, get the key from list of countries, get the value from summary.Countries.country dict
countryCases={}
for countryObj in listofCountries:
    for countryObjFromSummary in summary['Countries']:
        print(countryObjFromSummary["Country"])
        print(countryObjFromSummary['CountryCode'])
        print(countryObjFromSummary['TotalConfirmed'])
        if countryObj["ISO2"] == countryObjFromSummary['CountryCode']:
            print(countryObjFromSummary['TotalConfirmed'])
            countryCases[countryObj['Country']]=countryObjFromSummary['TotalConfirmed']
            json_data = json.dumps(countryCases)

print(countryCases)     
####

byCountry = "https://api.covid19api.com/country/india/status/confirmed?from=2020-03-01T00:00:00Z&to=2020-04-01T00:00:00Z"
payload1={}
headers1 = {}
output1 = requests.request("GET",byCountry , headers=headers1, data=payload1).json()
print(output1.)

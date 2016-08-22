import requests, httplib2, json, time, datetime
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

# This is the version used by Xinran Wang

class subscribers:
    values=[]
    
    def __init__(self):
        SCOPE = ["https://spreadsheets.google.com/feeds"]
        SECRETS_FILE = "AlertingService-879d85ad058f.json"
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

        spreadsheetId = '11kQlBCKJ_-NsQT6KYUIKo5hPRpkpM6UyRdJnxfVdJ4g'
        rangeName = 'Form Responses'#!A2:E'
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheetId, range=rangeName).execute()
        self.values = result.get('values', [])
        return
    
    def getTests(self):
        ret=[]
        testColumns=[4,6,7,8,10,11]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            for col,val in enumerate(row):
                if col not in testColumns: continue
                if val is None: continue
                if len(val)<1: continue
                tests = val.split(',')
                for test in tests:
                    test=test.strip()
                    if test not in ret:
                        ret.append(test)
        return ret

    def getSubscribers(self, testname):
        ret=[]
        testColumns=[4,6,7,8,10,11]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            for col,val in enumerate(row):
                if col not in testColumns: continue
                if val is None: continue
                if len(val)<1: continue
                tests = val.split(',')
                for test in tests:
                    test=test.strip()
                    if test==testname:
                        ret.append([row[3],row[1],row[2]]) #name, email, link
        return ret
    
    def getSubscribers_withSiteName(self, testname):
        ret=[]
        testColumns=[4,6,7,8,10,11]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            for col,val in enumerate(row):
                if col not in testColumns: continue
                if val is None: continue
                if len(val)<1: continue
                tests = val.split(',')
                for test in tests:
                    test=test.strip()
                    if test==testname:
                        ret.append([row[3],row[1],row[2],row[5]]) # name, email, link, sitename
        return ret

    def getUserInfo(self):
        ret=[]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            timestamp = row[0]
            timestamp = datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S')
            now = datetime.datetime.now()
            delta = now - timestamp
            threshold = datetime.timedelta(hours=1)
            freshuser = ( delta < threshold )  # user registered or updated settings in the past one hour
            ret.append([row[1], row[3], row[2], freshuser])  # email, name, link, boolean value indicating fresh user or not
        return ret
    
    def getAllUserBasicInfo(self):
        ret=[]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            ret.append([row[1], row[3], row[2]])  # email, name, link
        return ret

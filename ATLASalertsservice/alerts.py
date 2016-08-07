import requests, httplib2, json, time
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools


class subscribers:
    values=[]
    
    def __init__(self):
        SCOPE = ["https://spreadsheets.google.com/feeds"]
        SECRETS_FILE = "AlertingService-879d85ad058f.json"
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

        spreadsheetId = '19bS4cxqBEwr_cnCEfAkbaLo9nCLjWnTnhQHsZGK9TYU'
        #rangeName = 'Form Responses'#!A2:E'
        #result = service.spreadsheets().values().get(spreadsheetId=spreadsheetId, range=rangeName).execute()
        #self.values = result.get('values', [])
        return
    

    def addAlert(self, email, body):
        ret=[]
        for row in self.values:
            if row[0]=='Timestamp' : continue
            ret.append([row[3],row[1],row[2]]) #name, email, link
        return ret

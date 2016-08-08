import requests, httplib2, json, time
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools


class alerts:
    
    def __init__(self):
        SCOPE = ["https://spreadsheets.google.com/feeds"]
        SECRETS_FILE = "AlertingService-879d85ad058f.json"
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        self.service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)
        return
    

    def addAlert(self, test, email, text):
        spreadsheetId = '19bS4cxqBEwr_cnCEfAkbaLo9nCLjWnTnhQHsZGK9TYU'
        rangeName = test + '!A1:C1'
        myBody = {u'range': rangeName, u'values': [[time.strftime("%Y/%m/%d %H:%M:%S"), email, text]], u'majorDimension': u'ROWS'}
        cells = self.service.spreadsheets().values().append(spreadsheetId=spreadsheetId, range=rangeName,valueInputOption='RAW', insertDataOption='INSERT_ROWS', body=myBody).execute()
        return 

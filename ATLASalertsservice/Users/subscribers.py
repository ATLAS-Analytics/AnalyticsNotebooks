import requests, httplib2, json, time
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

class user:
    def __init__(self, email, name):
        self.email = email
        self.name  = name
        self.link  = None
        self.jobs= []
        self.intervals = []
        self.comparisons = False
    def set_response_edit_link(self, link):
        self.link  = link
    def set_rucio_name(self, username):
        self.rucio_username  = username
    def set_jobs(self, jobs_text):
        self.jobs  = [x.strip() for x in jobs_text.split(',')]
    def set_interval(self, interval_response):
        self.intervals  = [x.strip() for x in interval_response.split(',')]
    def add_comparisons(self):
        self.comparisons=True
    def to_string(self):
        return "user name:" + self.name + "  email:" + self.email + "  rucio:" + self.rucio_username
    
class subscribers:
    
    def __init__(self):
        SCOPE = ["https://spreadsheets.google.com/feeds"]
        SECRETS_FILE = "AlertingService-879d85ad058f.json"
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

        spreadsheetId = '114sgLyCSNkl7p-GAomLT1FgwYQy9fIVjznL3r9D5o8Q'
        rangeName = 'Form responses'#!A2:E'
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheetId, range=rangeName).execute()
        self.values = result.get('values', [])
        self.header = self.values[0]
        self.datarows = self.values[1:]
        self.load_users()
        return
    
    def load_users(self):
        """ a part of initialization, should not be called directly """
        self.users = []
        for nr, row in enumerate(self.datarows):
            u = user(self.get_cell_value(nr,'Email address'), self.get_cell_value(nr,'Your name'))
            u.set_response_edit_link(self.get_cell_value(nr,'ResponseEditLink'))
            u.set_rucio_name(self.get_cell_value(nr,'Your RUCIO username'))
            u.set_jobs(self.get_cell_value(nr,'Section'))
            u.set_interval(self.get_cell_value(nr,'Report frequency'))
            if self.get_cell_value(nr,'Report frequency')=='Yes':
                u.add_comparisons()
            self.users.append(u) 
        
    def get_cell_value(self, row_number, col_name):
        """ returns value of a cell in the spreadsheet """
        col_index = self.header.index(col_name)
        return self.datarows[row_number][col_index]
    
    def get_all_users(self):
        """ returns a list with all the users """
        return self.users
    
    def get_user(self, email):
        """ returns a user having a certain email """
        for u in self.users:
            if u.email==email:
                return u
    
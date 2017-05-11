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
        self.alerts= []
        self.immediate_subscriptions = []
        self.summary_subscriptions = []
    def set_response_edit_link(self, link):
        self.link  = link
    def set_site(self, site_text):
        self.sites  = [x.strip() for x in site_text.split(',')]
    def set_queue(self, queue_text):
        self.queues  = [x.strip() for x in queue_text.split(',')]
    def add_immediate_subscription(self, subscription_response):
        self.immediate_subscriptions.append(subscription_response)
    def add_summary_subscription(self, subscription_response):
        self.summary_subscriptions.append(subscription_response)
    def to_string(self):
        return "user name:" + self.name + "  email:" + self.email
    
class subscribers:
    
    def __init__(self):
        SCOPE = ["https://spreadsheets.google.com/feeds"]
        SECRETS_FILE = "AlertingService-879d85ad058f.json"
        credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

        spreadsheetId = '1yGCC2flsRL1Kfw5sxI9tO1gD_fKCdHgKqo5A9KD3w78'
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
            u.set_site(self.get_cell_value(nr,'Name of ATLAS site'))
            u.set_queue(self.get_cell_value(nr,'Queue name(s)'))
            for ncol, cell_value in enumerate(row):
                if cell_value=="Immediate": u.add_immediate_subscription(self.header[ncol])
                if cell_value=="Summary": u.add_summary_subscription(self.header[ncol])
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
        return
    
    def get_immediate_subscribers(self, alert_name):
        """ returns a list of all subscribers to the immediate alert. """
        ret=[]
        for u in self.users:
            if alert_name in u.immediate_subscriptions:
                ret.append(u)
        return ret

    def get_summary_subscribers(self, alert_name):
        """ returns a list of all subscribers to the summary alert. """
        ret=[]
        for u in self.users:
            if alert_name in u.summary_subscriptions:
                ret.append(u)
        return ret
    

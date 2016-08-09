import requests, httplib2, json, time
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools

SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
SECRETS_FILE = "AlertingService-879d85ad058f.json"
credentials = ServiceAccountCredentials.from_json_keyfile_name(SECRETS_FILE, SCOPE)
http = credentials.authorize(httplib2.Http())
service = discovery.build('gmail', 'v1', http=http)

        

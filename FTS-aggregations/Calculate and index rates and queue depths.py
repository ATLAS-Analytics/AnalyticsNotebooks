import requests
import pickle
import re
import datetime,time, sys
from elasticsearch import Elasticsearch, helpers, exceptions as es_exceptions
from elasticsearch.helpers import scan
import datetime as dt

import numpy as np
import pandas as pd

es = Elasticsearch([{'host':'atlas-kibana.mwt2.org', 'port':9200}],timeout=60)

days_around=5

date_to_process = sys.argv[1].split('-')
cdt = datetime.datetime(int(date_to_process[0]),int(date_to_process[1]),int(date_to_process[2]))
#cdt = datetime.datetime.utcnow() - datetime.timedelta(days=2) # to make sure data is in HDFS


# usefull functions

def store(docs_to_store):
    try:
       res = helpers.bulk(es, docs_to_store, raise_on_exception=True,request_timeout=60)
       #print("inserted:",res[0], '\tErrors:',res[1])
    except es_exceptions.ConnectionError as e:
       print('ConnectionError ', e)
    except es_exceptions.TransportError as e:
       print('TransportError ', e)
    except helpers.BulkIndexError as e:
       print(e[0])
       for i in e[1]:
          print(i)
    except Exception as e:
       print('Something seriously wrong happened.',e)


# #### Create structures to hold the data. Time bins are 1 minute.


period_start = cdt.replace(hour=0).replace(minute=0).replace(second=0)
period_end   = cdt.replace(hour=23).replace(minute=59).replace(second=59)

bot = dt.datetime(1970,1,1)
l_index_name='links_traffic_' + str(period_start.year) + '-' + str(period_start.month)# + '-' + str(period_start.day)
s_index_name='sites_traffic_' + str(period_start.year) + '-' + str(period_start.month)# + '-' + str(period_start.day)

ps = int((period_start - bot).total_seconds())
pe = int((period_end - bot).total_seconds())
print(period_start, period_end)

#caching periods
psb=int(ps/60)
pse=int(pe/60)
bins=[]

for t in range(psb,pse):
    bins.append(t*60)

endpoint_features = [ 'EndpointEgress', 'EndpointIgress', 'OutcomingTransfers', 'IncomingTransfers']
activities = [
    'Data_Consolidation', 'Production_Input', 'Data_Rebalancing', 'Production_Output', 'User_Subscriptions',
    'Data_Brokering', 'Express' 
    ]

class link:
    def __init__(self, src, dest):
        self.src=src
        self.dest=dest
        self.df=pd.DataFrame(0, index = bins, columns = [ 'rate' ]  + activities )
        self.df['rate']=self.df['rate'].astype('float64')
    def add_transfer(self, start_time, end_time, rate):
        st=int(start_time/60)
        et=int(end_time/60)
        area_covered = (et - st + 1) * 60 # area that will be covered in seconds
        actual_seconds = end_time - start_time
        scaled_rate = rate * actual_seconds / area_covered
        for ts in range(st,et+1):
            if ts>=psb and ts<pse: 
                timestamp = ts * 60
                val = self.df.get_value(timestamp, 'rate') + scaled_rate
                self.df.set_value(timestamp, 'rate', val)
    def add_queue(self, start_time, end_time, activity):
        if activity not in activities: return
        st=int(start_time/60)
        et=int(end_time/60)
        for ts in range(st,et+1):
            if ts>=psb and ts<pse: 
                timestamp = ts * 60
                val = self.df.get_value(timestamp, activity) + 1
                self.df.set_value(timestamp, activity, val)
    def stats(self):
        print(self.df.describe())
    def get_json_docs(self):
        docs=[]
        for index, row in self.df.iterrows():
            doc = {
                '_index': l_index_name,
                '_type' : 'docs',
                'timestamp' : int(index*1000),
                'src' : self.src,
                'dest' : self.dest,
                'rate' : int(row['rate'])
            }
            for activity in activities:
                if int(row[activity])>0: doc[activity]=int(row[activity])
            docs.append(doc)
        return docs
    
class endpoint:
    def __init__(self, name):
        self.name=name
        self.df=pd.DataFrame(0, index = bins, columns = endpoint_features)
        self.df['EndpointIgress'] = self.df['EndpointIgress'].astype('float64')
        self.df['EndpointEgress'] = self.df['EndpointEgress'].astype('float64')
    def add_transfer(self, start_time, end_time, rate, direction):
        st=int(start_time/60)
        et=int(end_time/60)
        area_covered = (et-st+1)*60 # area that will be covered in seconds
        actual_seconds = end_time - start_time
        scaled_rate = rate * actual_seconds / area_covered
        if direction: 
            drct='EndpointIgress'
            drct1='IncomingTransfers'
        else:
            drct='EndpointEgress'
            drct1='OutcomingTransfers'
        for ts in range(st,et+1):
            if ts>=psb and ts<pse: 
                timestamp = ts * 60
                val = self.df.get_value(timestamp, drct) + scaled_rate
                self.df.set_value(timestamp, drct, val)
                val = self.df.get_value(timestamp, drct1) + 1
                self.df.set_value(timestamp, drct1, val)
    def stats(self):
        print(self.df.describe())
    def get_json_docs(self):
        docs=[]
        for index, row in self.df.iterrows():
            docs.append({ 
                '_index': s_index_name,
                '_type' : 'docs',
                'timestamp' : int(index*1000),
                'name' : self.name,
                'ingress' : float(row['EndpointIgress']),
                'egress' : float(row['EndpointEgress']),
                'incoming' : int(row['IncomingTransfers']),
                'outcoming' : int(row['OutcomingTransfers'])
                }
            )
        return docs


# #### Load the data

query = {
    "size": 0,
    "_source": ["metadata.src_site", "metadata.dst_site", "metadata.activity","f_size",
                "processing_start","transfer_start","transfer_stop","processing_stop"],
    "query":{ 
        "bool" : {
            "must" : [
               # {"term" : { "src_rse" : "BNL-OSG2_DATADISK" }},
               # {"term" : { "dst_rse" : "CERN-PROD_DATADISK" }},
                {"term" : { "vo" : "atlas" }},
                {"term" : { "final_transfer_state" : "Ok"}},
                {"range" : {"processing_start" : {  "gte": period_start } }},
                {"range" : {"processing_stop" :   {  "lt" : period_end } }}
                ]
        }
    }        
}

scroll = scan(client=es, index="fts", query=query, scroll='5m', timeout="5m", size=10000)

endpoints={}
links={}

count = 0
for res in scroll:
    count += 1
#    print(res)
#    if count>10: break
    if not count%100000 : 
        print (count)
    r    = res['_source']
    if not ('src_site' in r['metadata'] and 'dst_site' in r['metadata']): continue
    src  = r['metadata']['src_site']
    dest = r['metadata']['dst_site']
    subm = r['processing_start']/1000
    star = r['transfer_start']/1000
    tran = r['transfer_stop']/1000
    transfer_duration = tran - star
    if transfer_duration > 0:
        rate = float(r['f_size']) / transfer_duration * 0.000000953674316
    
    if src not in endpoints: endpoints[src]=endpoint(src)
    if dest not in endpoints: endpoints[dest]=endpoint(dest)
    
    link_name = src + '->' + dest
    if link_name not in links: links[link_name]=link(src,dest)
    
    links[link_name].add_transfer( star, tran, rate) 
    links[link_name].add_queue( subm, star, r['metadata']['activity'].replace(' ','_'))
    
    endpoints[src].add_transfer( star, tran, rate, 0)  
    endpoints[dest].add_transfer( star, tran, rate, 1)  
      
print("docs read:", count)        
    #print(r['submitted_at'],r['started_at'],r['transferred_at'])


print('links:',len(links), '\tendpoints:',len(endpoints))

#print(links.keys())
#print(endpoints.keys())

#links['BNL-ATLAS->CERN-PROD'].df['rate']
#links['BNL-ATLAS->CERN-PROD'].stats()
#endpoints['CERN-PROD'].stats()

tp=int(len(links)/20)
for nl,link in enumerate(links.values()):
    if not nl%tp: print(nl," links indexed" )
    #print(link.get_json_docs())
    store(link.get_json_docs())
    #break
for endpoint in endpoints.values():
    print('endpoint indexed: ', endpoint.name)
    #print(endpoint.get_json_docs())
    store(endpoint.get_json_docs())
    #break


print('done')


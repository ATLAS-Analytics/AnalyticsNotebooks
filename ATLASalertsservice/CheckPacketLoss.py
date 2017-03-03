#!/home/ivukotic/anaconda3/bin/python
from elasticsearch import Elasticsearch, exceptions as es_exceptions, helpers
import sys
import datetime

cdt = datetime.datetime.utcnow()
cdt = datetime.datetime(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),0,0)

GT = (cdt - datetime.timedelta(hours=3)).strftime("%Y%m%dT%H%m%S+0000")
LT = cdt.strftime("%Y%m%dT%H%m%S+0000")
print('between: ', GT, ' and ', LT)

es = Elasticsearch(hosts=[{'host':'atlas-kibana.mwt2.org', 'port':9200}],timeout=60)

ipSite={}  # for mapping IP to Site name
toAlertOn=[]

def generate_doc(src_site_ip, dest_site_ip, measurements, avgpl):
   if src_site_ip not in ipSite:
        print('serious source mapping issue')
        return
   if dest_site_ip not in ipSite:
        print('serious destination mapping issue')
        return
   
   doc = {
        '_index':get_index_name(),
        '_type' : 'packetloss',
        'src' : src_site_ip,
        'dest' : dest_site_ip,
        'srcSite' : ipSite[src_site_ip],
        'destSite' : ipSite[dest_site_ip],
        'alarmTime' : int( (cdt-datetime.datetime(1970,1,1) ).total_seconds() * 1000 ),
        'measurements' : measurements,
        'packetLossAvg' : avgpl
   }
   return doc

def get_index_name():
    date = cdt.strftime("%Y-%m")   # date format is yyyy-mm
    index_name = 'alarms-'+date
    return index_name

indices = es.cat.indices(index="network_weather-*", h="index", request_timeout=600).split('\n')
indices = [x for x in indices if x != '']
indices = [x.strip() for x in indices]

# searching through last three indices as this simplifies utc issues
pday  = cdt - datetime.timedelta(days=1)
nday  = cdt + datetime.timedelta(days=1)
ind1 = 'network_weather-%d.%d.%d' % (cdt.year, cdt.month, cdt.day)
ind2 = 'network_weather-%d.%d.%d' % (pday.year, pday.month, pday.day)
ind3 = 'network_weather-%d.%d.%d' % (nday.year, nday.month, nday.day)

ind=[]
if ind1 in indices :
   ind.append(ind1)

if ind2 in indices:
   ind.append(ind2)

if ind3 in indices:
   ind.append(ind3)

if len(ind)==0:
   print ('no current indices found. Aborting.')
   sys.exit(1)
else:
   print('will use indices:', ind)

query={
   "size": 0,
   "query": {
    "bool": {
      "must": [
        {"term": { "_type" : "packet_loss_rate"}},
        {"term": { "srcProduction" : True }},
        {"term": { "destProduction" : True }}
      ],
      "filter" : {
        "range" : {
          "timestamp" :{ "gt" : GT, "lt" : LT }
        }
      }
    }
   },
    "aggs" : {
      "src" : {
        "terms" : { "field" : "src", "size": 1000 },
        "aggs" : {
          "dest" : {
            "terms" : {"field" : "dest", "size": 1000},
            "aggs" : {
              "avgpl" : {
                "avg" :{
                  "field" : "packet_loss"
              }
            }
          }
        }
      }
    },
    "srcSites" : {
      "terms" : { "field" : "src", "size": 1000 },
        "aggs" : {
          "srcsitename" : {
            "terms" : { "field" : "srcSite" }
        }
      }
    },
    "destSites" : {
      "terms" : { "field" : "dest", "size": 1000 },
        "aggs" : {
          "destsitename" : {
            "terms" : { "field" : "destSite" }
        }
      }
    }
  }
}

#execute query
res = es.search(index=ind, body=query, request_timeout=120)
#print(res)

srcsites=res['aggregations']['srcSites']['buckets']
#print(srcsites)
for sS in srcsites:
   #print(sS)
   siteName=sS['srcsitename']['buckets']
   if len(siteName)==0:
      siteName='UnknownSite'
   else:
      siteName=siteName[0]['key']
   ipSite[sS['key']]=siteName

destsites=res['aggregations']['destSites']['buckets']
#print(destsites)
for dS in destsites:
   #print(dS)
   siteName=dS['destsitename']['buckets']
   if len(siteName)==0:
      siteName='UnknownSite'
   else:
      siteName=siteName[0]['key']
   ipSite[dS['key']]=siteName

print(ipSite)


src=res['aggregations']['src']['buckets']
#print(src)

for s in src:
   #print(s)
   source=s['key']
   for d in s['dest']['buckets']:
      destination=d['key']
      avgpl=d['avgpl']['value']
      docs=d['doc_count']
#      print(source, destination, docs, avgpl)
      if avgpl > 0.02 and docs > 4:
         toAlertOn.append(generate_doc(source, destination, docs, avgpl))

for alert in toAlertOn:
   print(alert)

try:
   res = helpers.bulk(es, toAlertOn, raise_on_exception=True,request_timeout=60)
   print("inserted:",res[0], '\tErrors:',res[1])
except es_exceptions.ConnectionError as e:
   print('ConnectionError ', e)
except es_exceptions.TransportError as e:
   print('TransportError ', e)
except helpers.BulkIndexError as e:
   print(e[0])
   for i in e[1]:
      print(i)
except:
   print('Something seriously wrong happened.')


class es_query:
    
   def __init__(self):
      return

   def setquery(self, condition, inittime, endtime):
      self.qtxt = {
         "size": 10000,
         "query": {
            "bool": {
               "must": [
                  {
                     "query_string": {
                        "query": condition,
                        "analyze_wildcard": True,
                        "lowercase_expanded_terms": False
                     }
                  },
                  {
                     "range": {
                        "@timestamp": {
                           "gte": inittime,
                           "lte": endtime,
                           "format": "basic_date_time"
                        }
                     }
                  }
               ],
               "must_not": []
            }
         }
      }
      return self.qtxt

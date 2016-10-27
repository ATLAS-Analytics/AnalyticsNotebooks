class equivalence:
   def __init__(self, position, path1):
      self.before = path1.hops[position-1]
      self.after  = path1.hops[position+1]
      self.eqIPs = {}
      self.totalOccurences = 0
   def addPair(self, position, path1, path2):
      IP1 = path1.hops[position]
      IP2 = path2.hops[position]
      if IP1 not in self.eqIPs: self.eqIPs[IP1]=0
      if IP2 not in self.eqIPs: self.eqIPs[IP2]=0
      self.eqIPs[IP1] += path1.occurences
      self.eqIPs[IP2] += path2.occurences
      self.totalOccurences += (path1.occurences + path2.occurences)
   def prnt(self):
      for ip,count in self.eqIPs.items():
         print ("ip:",ip, "count:",count, count/self.totalOccurences*100,"%")
      print ('-------------------------------------------------') 

class equivalences:
   def __init__(self):
      self.eqs=[]
   def addEquivalence(self, position, path1, path2):
      before = path1.hops[position-1]
      after  = path1.hops[position+1]
      for eq in self.eqs:
         if before == eq.before and after == eq.after: 
            eq.addPair(position, path1, path2)
            return
      eq = equivalence(position, path1)
      eq.addPair(position, path1, path2)
      self.eqs.append(eq)
   def prnt(self):
      if len(self.eqs) == 0: return
      print('equivalences:',len(self.eqs))
      for e in self.eqs:
         e.prnt()
      
class path:
   def __init__(self, h, hops, occurences=0, rtts=[]):
      self.hash=h
      self.hops=hops
      self.occurences=occurences
      self.rtts=rtts
   def checkSingleNone(self):
      yes=0
      self.singleNone=False
      for h in self.hops[:-1]:
         if h==None and yes==0: yes=1
         if h!=None and yes==1: 
            self.singleNone=True
           #print ('random none - ', self.hops)
            break
   def getIPs(self):
      return set(self.hops)
   def prnt(self):
      print('seen:', self.occurences, '\trandom None:', self.singleNone, '\thops:', self.hops, '\trtts:', self.rtts)

class link:
   def __init__(self, source, destination): 
      self.source=source
      self.destination=destination
      self.paths={}
      self.equs=equivalences()
   def addPath(self, path):
      if path.hash not in self.paths:
         path.checkSingleNone()
         path.hops=[self.source]+path.hops
         if path.hops[-1]==None: path.hops[-1]=self.destination
         self.paths[path.hash]=path
      else:
         print('already have that path')
   def getNpaths(self):
       return len(self.paths)
   def getNtests(self):
       s = 0
       for p in self.paths:
          s += self.paths[p].occurences
       return s
   def getIPs(self):
       IPs=set()
       for p in self.paths.values():
          IPs=IPs.union(p.getIPs())
       return IPs
   def findEQs(self):
       pths=list(self.paths.values())
       l = len(pths)
       for i in range(l):
          for j in range(0,i):
             p1=pths[i].hops
             p2=pths[j].hops
             if len(p1) != len(p2): continue
             for c in range(1,len(p1)-1):
                if p1[c]==p2[c]:continue
                if p1[c]==None or p2[c]==None: continue
                if p1[c-1]==None or p2[c-1]==None: continue
                if p1[c+1]==None or p2[c+1]==None: continue
                if p1[c-1]!=p2[c-1] or p1[c+1]!=p2[c+1]: continue
                self.equs.addEquivalence(c,pths[i],pths[j])
                print (c)
                pths[i].prnt()
                pths[j].prnt()
   def nSingleNones(self):
      s=0
      for p in self.paths:
         if self.paths[p].singleNone: s += 1
      return s
   def prnt(self):
      print('src: ', self.source, '\tdest:', self.destination, '\ttests:',self.getNtests(), '\tpaths:', self.getNpaths(), '\trandomNones:',self.nSingleNones(), '\tIPs:', len(self.getIPs()))
      for p in self.paths:
         self.paths[p].prnt()
class mesh:
   def __init__(self):
      self.links=[]
   def addLink(self, source, destination):
      found=False
      for l in self.links:
         if l.source==source and l.destination==destination:
            return l
      if not found:
         nLink=link(source,destination)
         self.links.append(nLink)
         return nLink
   def getNlinks(self):
      return len(self.links)
   def getIPs(self):
      IPs=set()
      for l in self.links:
         IPs=IPs.union(l.getIPs())
      return IPs 

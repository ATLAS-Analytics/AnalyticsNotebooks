import subscribers, alerts

S = subscribers.subscribers()

testName = 'Cluster is in RED'
subscribersToRed =  S.getSubscribers(testName)

for s in subscribersToRed:
    print(s)
print('\n\n')
for s in S.getSubscribers_withSiteName('Packet loss increase for link(s) where your site is a source or destination'):
    print(s)

print (S.getAllUserBasicInfo())

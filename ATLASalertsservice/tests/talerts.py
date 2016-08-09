from email.mime.text import MIMEText
from subprocess import Popen, PIPE

body='test failined.a.s...'

def sendMail(test,to,body):
   msg = MIMEText(body)
   msg['Subject'] = 'The test title' 
   msg['From'] = 'AAAS@mwt2.org'
   msg['To'] = to

   p = Popen(["/usr/sbin/sendmail", "-t", "-oi", "-r AAAS@mwt2.org"], stdin=PIPE)
   print(msg.as_string())
   p.communicate(msg.as_string().encode('utf-8'))

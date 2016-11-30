import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

import time



def SimpleEmail(msg, who=None):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login("cluster.peter@gmail.com","cluster2610")
	toaddr = "peter.naylor@mines-paristech.fr" if who is None else who
	server.sendmail("cluster.peter@gmail.com", toaddr, msg)
	server.quit()



def ElaborateEmail(body, subject):
	fromaddr = "cluster.peter@gmail.com"
	toaddr = "peter.naylor@mines-paristech.fr"
	msg = MIMEMultipart()
	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Subject'] = subject
	 
	msg.attach(MIMEText(body, 'plain'))
	text = msg.as_string()
	SimpleEmail(text)

def EmailSpam(body, who, repeat = 5):
	fromaddr = "cluster.peter@gmail.com"
	toaddr = who
	words = body.split(' ')
	for i in range(repeat):
		for word in words:
			msg = MIMEMultipart()
			msg['From'] = fromaddr
			msg['To'] = toaddr
			msg['Subject'] = "this is not a spam martin"
			msg.attach(MIMEText(word, 'plain'))
			text = msg.as_string()
			SimpleEmail(text)
			time.sleep(120)
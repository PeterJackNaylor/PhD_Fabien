import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText




def SimpleEmail(msg):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login("cluster.peter@gmail.com","cluster2610")
	server.sendmail("cluster.peter@gmail.com", "peter.naylor@mines-paristech.fr", msg)
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
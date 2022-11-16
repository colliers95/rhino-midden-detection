#!/usr/bin/env python3
# Send emails and read their responses

from smtplib import SMTP_SSL, SMTP_SSL_PORT
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.encoders import encode_base64

# Get credentials
with open("email.config") as config:
    creds = {
        line.strip().split("|")[0]: line.strip().split("|")[1]
        for line in config.readlines()
    }

# Update to change sender (unlikely) and target (likely)
from_email = creds.get("sender_email")
to_emails = [
    creds.get("sender_email"),
]


def send_email(
    from_email, from_password, to_emails, subject_line, body_text, image_paths
):
    # https://www.devdungeon.com/content/read-and-send-email-python#toc-2

    # Create multipart MIME email
    email_message = MIMEMultipart()
    email_message.add_header("To", ", ".join(to_emails))
    email_message.add_header("From", from_email)
    email_message.add_header("Subject", subject_line)

    # Create text and HTML bodies for email
    text_part = MIMEText(body_text, "plain")
    email_message.attach(text_part)

    # Create image attachment
    for file in image_paths:
        with open(file, "rb") as fp:
            img_data = fp.read()
            email_message.add_attachment(
                img_data,
                maintype="image",
            )

    # Connect, authenticate, and send mail
    smtp_server = SMTP_SSL("smtp.gmail.com", port=SMTP_SSL_PORT)
    smtp_server.set_debuglevel(1)  # Show SMTP server interactions
    smtp_server.login(from_email, from_password)
    smtp_server.sendmail(from_email, to_emails, email_message.as_bytes())

    # Disconnect
    smtp_server.quit()

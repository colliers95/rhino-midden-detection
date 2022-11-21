#!/usr/bin/env python3
# Send emails and read their responses

from smtplib import SMTP_SSL
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List
from ssl import create_default_context
import os
import zipfile

# Get credentials
with open("email.config") as config:
    creds = {
        line.strip().split("|")[0]: line.strip().split("|")[1]
        for line in config.readlines()
    }

# Update to change sender (unlikely) and target (likely)
sender_email = creds.get("sender_email")
sender_password = creds.get("sender_password")
receiver_emails = [
    creds.get("sender_email"),
]


class SendReceiveEmail:
    def __init__(
        self,
        from_email: str,
        from_password: str,
        to_emails: List[str],
    ):
        self.from_email = from_email
        self.from_password = from_password
        self.to_emails = to_emails
        self.subject_line = ""
        self.body_text = ""
        self.body_html = ""

    def send_email(
        self,
        subject_line: str,
        body_text: str,
        body_html: str,
        image_paths: List[str] = ["None"],
    ):
        # https://www.devdungeon.com/content/read-and-send-email-python#toc-2

        # Create multipart MIME email
        email_message = MIMEMultipart()
        email_message.add_header("To", ", ".join(self.to_emails))
        email_message.add_header("From", self.from_email)
        email_message.add_header("Subject", subject_line)

        # Create text and HTML bodies for email
        text_part = MIMEText(body_text, "plain")
        html_part = MIMEText(body_html, "html")
        email_message.attach(text_part)
        email_message.attach(html_part)

        if not all([path == "None" for path in image_paths]):
            # Create zipfile attachment full of imaes
            with zipfile.ZipFile("midden_images.zip", "w") as img_zip:
                for filepath in image_paths:
                    with open(filepath, "rb") as fp:
                        img_data = fp.read()
                    img_zip.writestr(os.path.basename(filepath), img_data)
            # Add zip to email
            msg = MIMEBase("application", "zip")
            zf = open("midden_images.zip", "rb")
            msg.set_payload(zf.read())
            encoders.encode_base64(msg)
            msg.add_header(
                "Content-Disposition", "attachment", filename="midden_images.zip"
            )
            email_message.attach(msg)

        # Connect, authenticate, and send mail
        context = create_default_context()
        with SMTP_SSL("smtp.gmail.com", port=465, context=context) as smtp_server:
            smtp_server.set_debuglevel(1)  # Show SMTP server interactions
            smtp_server.login(from_email, from_password)
            smtp_server.sendmail(from_email, to_emails, email_message.as_string())
            # Disconnect
            smtp_server.quit()

        # Clean up
        os.remove("midden_images.zip")

    def create_email_parts(self):
        pass


if __name__ == "__main__":
    # Images to send decided by the network
    img_indices = ["001", "002"]

    image_paths = [
        [
            "./TestImage/" + "tile_{}_rgb.png".format(i),
            "./TestImage/" + "tile_{}_thermal.png".format(i),
        ]
        for i in img_indices
    ]
    image_paths = [item for sublist in image_paths for item in sublist]
    subject = "IMPORTANT: MIDDEN VERIFICATION FOR " + ", ".join(img_indices)
    textheader1 = "Please check the attached images to verify whether they contain middens. There is a single thermal and rgb tile for each index.\n"
    textheader2 = "Please send your response by clicking the following "
    htmlbody = '<a href="mailto:{0}?subject={1}&body={2}">link</a>'.format(
        sender_email, "test", "body test"
    )
    textbody = textheader1 + textheader2
    send_email(
        sender_email,
        sender_password,
        receiver_emails,
        subject,
        textbody,
        htmlbody,
        image_paths=image_paths,
    )

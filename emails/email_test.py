#!/usr/bin/env python3
# Send emails and read their responses

from smtplib import SMTP_SSL
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders, message_from_string, message_from_bytes, policy
from typing import List
from ssl import create_default_context
import os
import zipfile
import imaplib

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
    creds.get("sender_email"),  # coauthor_email
]


class SendReceiveEmail:
    def __init__(
        self,
        from_email: str,
        from_password: str,
        to_emails: List[str],
        img_indices: List[str],
        image_paths: List[str],
    ):
        self.from_email = from_email
        self.from_password = from_password
        self.to_emails = to_emails
        self.subject_line = ""
        self.body_text = ""
        self.body_html = ""
        self.img_indices = img_indices
        self.image_paths = image_paths
        self.email_answers = {}

    def get_email_answers(self):
        return self.email_answers

    def send_email(self):
        # https://www.devdungeon.com/content/read-and-send-email-python#toc-2

        # Create multipart MIME email
        email_message = MIMEMultipart()
        email_message.add_header("To", ", ".join(self.to_emails))
        email_message.add_header("From", self.from_email)
        email_message.add_header("Subject", self.subject_line)

        # Create text and HTML bodies for email
        top_text_part = MIMEText(self.top_body_text, "plain")
        bottom_text_part = MIMEText(self.bottom_body_text, "plain")
        html_part = MIMEText(self.body_html, "html")
        email_message.attach(top_text_part)
        email_message.attach(html_part)
        email_message.attach(bottom_text_part)

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
            # smtp_server.set_debuglevel(1)  # Show SMTP server interactions
            smtp_server.login(self.from_email, self.from_password)
            smtp_server.sendmail(
                self.from_email, self.to_emails, email_message.as_string()
            )
            # Disconnect
            smtp_server.quit()

        print("Email sent")

    def create_email_parts(self):
        self.subject_line = "IMPORTANT: MIDDEN VERIFICATION FOR " + ", ".join(
            self.img_indices
        )
        textheader1 = "Please check the attached images to verify whether they contain middens. There is a single thermal and rgb tile for each index.\n"
        textheader2 = "Please send your response by clicking this "
        textheader3 = ", editing nothing except for the yes/no answers\n"
        # Store creation time to find email later
        self.create_time = datetime.now().strftime("%m-%d-%y %H:%M:%S")
        self.reply_header = (
            "REPLY: " + ", ".join(self.img_indices) + " created " + self.create_time
        )
        reply_text = "\n\n".join([idx_str + ": no" for idx_str in img_indices])
        self.body_html = '<a href="mailto:{0}?subject={1}&body={2}">link</a>'.format(
            self.from_email, self.reply_header, reply_text
        )
        self.top_body_text = textheader1 + textheader2
        self.bottom_body_text = textheader3
        print("Email sections created")

    def read_email(self):
        # Connect and login to IMAP mail server
        imap_server = imaplib.IMAP4_SSL(host="imap.gmail.com")
        imap_server.login(self.from_email, self.from_password)

        # Choose the mailbox (folder) to search
        # Case sensitive!
        imap_server.select("INBOX")  # Default is `INBOX`

        # Search for emails in the mailbox that was selected.
        # First, you need to search and get the message IDs.
        # Then you can fetch specific messages with the IDs.
        # Search filters are explained in the RFC at:
        # https://tools.ietf.org/html/rfc3501#section-6.4.4
        search_criteria = 'HEADER Subject "{}"'.format(self.reply_header)
        charset = None  # All
        message_numbers = []
        while not message_numbers:  # Empty list if no replies found
            imap_server.noop()  # refres
            response_code, message_numbers_raw = imap_server.search(
                charset, search_criteria
            )
            message_numbers = message_numbers_raw[0].split()
        print(search_criteria)
        print(f"Search response: {response_code}")  # e.g. OK
        print(
            f"Message numbers: {message_numbers_raw}"
        )  # e.g. ['1 2'] A list, with string of message IDs

        # Fetch full message based on the message numbers obtained from search
        response_code, message_data = imap_server.fetch(message_numbers[0], "(RFC822)")
        print(f"Fetch response for message {message_numbers[0]}: {response_code}")
        print(f"Raw email data:\n{message_data}")

        try:
            msg = message_from_bytes(
                message_data[1][1], policy=policy.default
            )  # Google mail
        except IndexError:
            msg = message_from_bytes(
                message_data[0][1], policy=policy.default
            )  # Apple mail
        body = msg.get_body(("plain",))
        body = body.get_content()
        body_lines = body.split("\n")
        body_els = [el.replace("\r", "").split(":") for el in body_lines]
        answers = {lst[0]: lst[1].strip() for lst in body_els if lst != [""]}
        print(f"Extracted email body:\n{body}")

        self.email_answers = answers

        imap_server.close()
        imap_server.logout()


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
    mail = SendReceiveEmail(
        sender_email, sender_password, receiver_emails, img_indices, image_paths
    )
    mail.create_email_parts()
    mail.send_email()
    mail.read_email()
    email_answers = mail.get_email_answers()

    print(email_answers)

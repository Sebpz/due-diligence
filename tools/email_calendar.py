import os.path
import base64
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/calendar.events'
]

def authenticate_google_api():
    """
    Authenticates with Google APIs and returns service objects for Gmail and Calendar.
    
    Returns:
        tuple: (gmail_service, calendar_service) - The authenticated service objects
               or (None, None) if authentication fails
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_231535190646-or8nbgnmp286d7vcq9jdruk2hsm89m84.apps.googleusercontent.com.json', 
                SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Build the Gmail service
        gmail_service = build('gmail', 'v1', credentials=creds)
        # Build the Calendar service
        calendar_service = build('calendar', 'v3', credentials=creds)
        
        return gmail_service, calendar_service

    except Exception as e:
        print(f"Error creating service(s): {e}")
        return None, None

def draft_gmail_email(gmail_service, to: str, subject: str, body: str):
    """
    Creates a new draft email in the user's Gmail account.
    
    Args:
        gmail_service: The authenticated Gmail service object.
        to: The recipient's email address.
        subject: The subject of the email.
        body: The body content of the email.
    """
    try:
        message = EmailMessage()
        message.set_content(body)
        message['To'] = to
        message['Subject'] = subject

        # Encoded the message and create a 'message' body for the API.
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'message': {'raw': encoded_message}}

        # Make the API call to create the draft.
        draft = gmail_service.users().drafts().create(userId='me', body=create_message).execute()
        print(f"Draft ID: {draft['id']} created successfully.")
    
    except HttpError as error:
        print(f"An error occurred while drafting the email: {error}")

def create_google_calendar_event(calendar_service, summary: str, start_time: datetime, end_time: datetime, description: str):
    """
    Creates a new event on the user's primary Google Calendar.
    
    Args:
        calendar_service: The authenticated Google Calendar service object.
        summary: The title of the event.
        start_time: The starting datetime object for the event.
        end_time: The ending datetime object for the event.
        description: A description for the event.
    """
    try:
        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'America/Los_Angeles',  # Set your timezone here
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'America/Los_Angeles', # Set your timezone here
            },
        }

        # Make the API call to insert the event.
        event = calendar_service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")

    except HttpError as error:
        print(f"An error occurred while creating the calendar event: {error}")


# --- Example Usage ---
if __name__ == "__main__":
    gmail_service, calendar_service = authenticate_google_api()

    if gmail_service and calendar_service:
        # Example for drafting a Gmail email
        draft_gmail_email(
            gmail_service,
            to="sebastianporterzadro@gmail.com",
            subject="Follow-up on AI News",
            body="I found some interesting articles about AI. Let's discuss."
        )

        # Example for creating a Google Calendar event
        now = datetime.now()
        one_hour_from_now = now + timedelta(hours=1)
    
        create_google_calendar_event(
            calendar_service,
            summary="AI News Discussion",
            start_time=now,
            end_time=one_hour_from_now,
            description="A quick chat about the latest AI news articles we found."
        )

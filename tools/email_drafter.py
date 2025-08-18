import requests
import json

def draft_email_with_graph_api(
    access_token: str,
    subject: str,
    to_recipients: list,
    body_content: str
):
    """
    Drafts a new email message in the user's Outlook mailbox using the Microsoft Graph API.

    This function requires a valid access token with 'Mail.ReadWrite' permissions.

    Args:
        access_token: The OAuth 2.0 access token for authentication.
        subject: The subject line of the email.
        to_recipients: A list of recipient email addresses (e.g., ['jane.doe@example.com']).
        body_content: The body content of the email, which can be HTML.

    Returns:
        A tuple (bool, dict) indicating success and the API response JSON.
    """
    # The Microsoft Graph API endpoint for creating a message draft
    # 'me' refers to the user associated with the access token
    api_url = "https://graph.microsoft.com/v1.0/me/messages"

    # Define the headers, including the Authorization token
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Format the recipients for the API payload
    recipients = [{"emailAddress": {"address": email}} for email in to_recipients]

    # Construct the JSON payload for the email message
    # 'saveToSentItems': False is used here to explicitly create a draft
    # The 'body' content type is set to 'html' for rich text formatting
    payload = {
        "subject": subject,
        "body": {
            "contentType": "HTML",
            "content": body_content
        },
        "toRecipients": recipients,
        "isDraft": True
    }

    try:
        # Make the POST request to the Graph API to create the draft
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the JSON response
        response_data = response.json()
        print(f"Successfully drafted email with subject: '{subject}'")
        return True, response_data

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return False, {"error": e.response.text}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {"error": str(e)}

# --- Example Usage ---
# NOTE: Replace 'YOUR_ACCESS_TOKEN' with a valid token
# and 'your.email@example.com' with a real recipient.
if __name__ == "__main__":
    # Placeholder for the access token. You must acquire this token
    # through a proper OAuth 2.0 authentication flow.
    # A valid JWT token should be in format: header.payload.signature
    ACCESS_TOKEN = "eyJ0eXAiOi...97cb4c2c-7f61-41e6-8af2-9f4855733482...S03ekjF1" # Replace with valid JWT token

    # Define the email content
    EMAIL_SUBJECT = "Quarterly Portfolio Review Report"
    RECIPIENT_LIST = ["client.email@example.com"]
    EMAIL_BODY_HTML = """
    <html>
        <body>
            <p>Dear Client,</p>
            <p>I hope this email finds you well. Your quarterly portfolio review report is attached for your review.</p>
            <p>Please let me know if you would like to schedule a time to discuss it in more detail.</p>
            <p>Best regards,<br>
            Your Name</p>
        </body>
    </html>
    """

    # Call the function to draft the email
    success, result = draft_email_with_graph_api(
        access_token=ACCESS_TOKEN,
        subject=EMAIL_SUBJECT,
        to_recipients=RECIPIENT_LIST,
        body_content=EMAIL_BODY_HTML
    )

    if success:
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
    else:
        print("\nFailed to draft email.")

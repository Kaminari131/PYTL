import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# Phạm vi quyền hạn (scope) cần sử dụng
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_emails(service):
    """Retrieve unread emails."""
    try:
        results = service.users().messages().list(userId='me', q='is:unread').execute()
        messages = results.get('messages', [])
        emails = []
        if messages:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                subject = next(
                    (header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'),
                    "No Subject"
                )
                snippet = msg.get('snippet', "")
                emails.append((message['id'], subject, snippet))
        return emails
    except Exception as e:
        print(f"Error retrieving emails: {e}")
        return []

vectorizer = TfidfVectorizer()

def classify_email(subject, snippet, model, vectorizer):
    """Classify email as spam or not spam."""
    text = subject + " " + snippet
    if not hasattr(vectorizer, 'vocabulary_'):
        print("Vectorizer not fitted. Please fit the vectorizer with training data first.")
        return None
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]


# When calling classify_email:
def move_to_spam(service, message_id):
    """Move email to spam folder."""
    service.users().messages().modify(
        userId='me', id=message_id,
        body={'addLabelIds': ['SPAM'], 'removeLabelIds': ['INBOX']}
    ).execute()

def main():
    # Load the pre-trained model
    with open('randomforest.pkl', 'rb') as f:
        model = pickle.load(f)

    # Authenticate Gmail API
    service = authenticate_gmail()

    # Retrieve unread emails
    emails = get_emails(service)

    if not emails:
        print("No unread emails found.")
        return

    # Classify and handle emails
    for message_id, subject, snippet in emails:
        classification = classify_email(subject, snippet, model,vectorizer)
        if classification == 'spam':
            move_to_spam(service, message_id)
            print(f"Moved email '{subject}' to Spam.")
if __name__ == '__main__':
    main()

from twilio.rest import Client

account_sid = 'AC2dec79537f7670dbb88850a65986aa95'
auth_token = 'a36a491447e018bf84341b8cc7a6d416'
client = Client(account_sid, auth_token)

def send_msg(phone_number, full_name):
    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='Hello from Twilio!',
    to='whatsapp:+917975070214'
    )
    return message.sid
# print(message.sid)
print(send_msg("+917975070214", "nikhil"))
from requests import session
def login():
    url = 'http://localhost:8082/loginin?username=penggan&password=penggan'
    s = session()
    response = s.post(url)
    print(response.content)

def getComment():
    url = 'http://localhost:8082/getComment?news_id=64'
    header = {
        'uid': 'penggan',
        'Cookie': "JSESSIONID=7FCFC63BE4482FA839C3AB36653C16E3"
    }
    s = session()
    response = s.get(url, headers=header)
    print(response.content)
    print(response.headers['isAlive'])

getComment()
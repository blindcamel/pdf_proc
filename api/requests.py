import requests
files = {'file': open('./uploads/80114.pdf', 'rb')}
response = requests.post('https://pdf-process.fly.dev/upload/', files=files)
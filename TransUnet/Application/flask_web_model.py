import requests

resp = requests.post("http://127.0.0.1:5000/", files={"file": open('../Image_2/val/val-org-img/476.jpg', 'rb')})

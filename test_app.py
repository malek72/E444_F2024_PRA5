import unittest
import requests
import json

# Citation: Using help from Copilot
class TestSentimentApp(unittest.TestCase):

    def setUp(self):
        self.base_url = "http://serve-sentiment-env.eba-dhur7uxb.us-east-2.elasticbeanstalk.com/predict"
        self.headers = {"Content-Type": "application/json"}
    
    def fake_1(self):
        data = {"text": "This is fake news"}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "FAKE")

    def fake_2(self):
        data = {"text": "Breaking: Aliens have landed on Earth!"}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "FAKE")
    
    def real_1(self):
        data = {"text": "The government passed a new education reform today."}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "REAL")
    
    def real_2(self):
        data = {"text": "The government announced new infrastructure projects for the upcoming year."}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "REAL")

if __name__ == "__main__":
    unittest.main()

import unittest
import requests
import json

class TestSentimentApp(unittest.TestCase):

    def setUp(self):
        # URL of the deployed app
        self.base_url = "http://serve-sentiment-env.eba-dhur7uxb.us-east-2.elasticbeanstalk.com/predict"
        # Headers for the POST request
        self.headers = {"Content-Type": "application/json"}
    
    # Test case 1: Fake news input
    def test_fake_news_1(self):
        data = {"text": "This is fake news"}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "FAKE")

    # Test case 2: Another fake news input
    def test_fake_news_2(self):
        data = {"text": "Breaking: Aliens have landed on Earth!"}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "FAKE")
    
    # Test case 3: Real news input
    def test_real_news_1(self):
        data = {"text": "The government passed a new education reform today."}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "REAL")
    
    def test_real_news_2(self):
        # New input text that is likely to be classified as "REAL"
        data = {"text": "The government announced new infrastructure projects for the upcoming year."}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        result = response.json()
        self.assertEqual(result["prediction"], "REAL")

if __name__ == "__main__":
    unittest.main()

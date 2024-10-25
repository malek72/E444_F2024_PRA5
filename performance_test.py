import requests
import json
import time
import csv
import matplotlib.pyplot as plt
# Citation: Using help from Copilot
url = "http://serve-sentiment-env.eba-dhur7uxb.us-east-2.elasticbeanstalk.com/predict"

headers = {"Content-Type": "application/json"}

test_cases = [
    {"text": "This is fake news"},
    {"text": "Breaking: Aliens have landed on Earth!"},
    {"text": "The government passed a new education reform today."},
    {"text": "The stock market is performing well this quarter."}
]

num_requests_per_case = 100

results = []
latencies = [[] for _ in range(len(test_cases))]

for idx, test_case in enumerate(test_cases):
    for i in range(num_requests_per_case):
        start_time = time.time()
        response = requests.post(url, headers=headers, json=test_case)
        end_time = time.time()
        latency = end_time - start_time
        results.append({
            "Test Case": f"Test Case {idx + 1}",
            "Text": test_case["text"],
            "Request Number": i + 1,
            "Latency": latency
        })
        latencies[idx].append(latency)
        print(f"Test Case {idx + 1} - Request {i + 1} Latency: {latency:.4f} seconds")

with open("performance_results.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Test Case", "Text", "Request Number", "Latency"])
    writer.writeheader()
    writer.writerows(results)

print("\nPerformance results saved to performance_results.csv")

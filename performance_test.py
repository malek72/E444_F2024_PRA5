import requests
import json
import time
import csv
import matplotlib.pyplot as plt

# The URL of your deployed AWS Elastic Beanstalk server
url = "http://serve-sentiment-env.eba-dhur7uxb.us-east-2.elasticbeanstalk.com/predict"

# The headers for the POST request
headers = {"Content-Type": "application/json"}

# Test cases: two fake news and two real news examples
test_cases = [
    {"text": "This is fake news"},
    {"text": "Breaking: Aliens have landed on Earth!"},
    {"text": "The government passed a new education reform today."},
    {"text": "The stock market is performing well this quarter."}
]

# Number of API calls per test case
num_requests_per_case = 100

# Store results
results = []
latencies = [[] for _ in range(len(test_cases))]  # List to store latencies for each test case

# Loop over each test case
for idx, test_case in enumerate(test_cases):
    for i in range(num_requests_per_case):
        # Measure start time
        start_time = time.time()
        
        # Send the POST request
        response = requests.post(url, headers=headers, json=test_case)
        
        # Measure end time
        end_time = time.time()
        
        # Calculate the latency for this request
        latency = end_time - start_time
        results.append({
            "Test Case": f"Test Case {idx + 1}",
            "Text": test_case["text"],
            "Request Number": i + 1,
            "Latency": latency
        })
        
        # Append latency to the appropriate test case list
        latencies[idx].append(latency)
        
        # Print the current iteration and the latency
        print(f"Test Case {idx + 1} - Request {i + 1} Latency: {latency:.4f} seconds")

# Write the results to a CSV file
with open("performance_results.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Test Case", "Text", "Request Number", "Latency"])
    writer.writeheader()
    writer.writerows(results)

print("\nPerformance results saved to performance_results.csv")

# Plotting the box plot for each test case
plt.figure(figsize=(10, 6))
plt.ylim(0, 0.5)
plt.boxplot(latencies, labels=[f"Test Case {i+1}" for i in range(len(test_cases))])
plt.title("Latency Box Plot for Each Test Case")
plt.xlabel("Test Case")
plt.ylabel("Latency (seconds)")
plt.show()

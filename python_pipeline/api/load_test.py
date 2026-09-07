import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000/predict"

payload = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 8,
    "srv_count": 8,
    "serror_rate": 0,
    "srv_serror_rate": 0,
    "rerror_rate": 0,
    "srv_rerror_rate": 0,
    "same_srv_rate": 1,
    "diff_srv_rate": 0,
    "srv_diff_host_rate": 0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1,
    "dst_host_diff_srv_rate": 0,
    "dst_host_same_src_port_rate": 0.11,
    "dst_host_srv_diff_host_rate": 0,
    "dst_host_serror_rate": 0,
    "dst_host_srv_serror_rate": 0,
    "dst_host_rerror_rate": 0,
    "dst_host_srv_rerror_rate": 0,
}

data = json.dumps(payload).encode()

latencies = []

for i in range(100):
    request = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request) as response:
        response.read()

    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)

latencies.sort()

print(f"Requests: {len(latencies)}")
print(f"Mean:     {statistics.mean(latencies):.2f} ms")
print(f"Median:   {statistics.median(latencies):.2f} ms")
print(f"P95:      {latencies[94]:.2f} ms")
print(f"P99:      {latencies[98]:.2f} ms")
print(f"Min:      {latencies[0]:.2f} ms")
print(f"Max:      {latencies[-1]:.2f} ms")
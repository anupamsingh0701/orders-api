import os
import re
import time
import requests

def get_tunnel_url():
    log_file = "llm_tunnel.log"
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None
    matches = re.findall(r'https?://[a-zA-Z0-9.-]+\.lhr\.life', content)
    if matches:
        return matches[-1]
    return None

def test_endpoint(url):
    print(f"\n--- Testing Endpoint: {url} ---")
    headers = {
        "Content-Type": "application/json"
    }
    
    # 1. Test Echo Test
    payload_echo = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Please repeat this token: TKJWIUGJ"}],
        "stream": False
    }
    try:
        r = requests.post(url, json=payload_echo, headers=headers, timeout=10)
        print("Echo test status:", r.status_code)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print("Echo response:", content)
        if "tkjwiugj" in content.lower():
            print("Echo test: SUCCESS")
        else:
            print("Echo test: FAILED")
    except Exception as e:
        print("Echo test: ERROR -", e)

    # 2. Test Arithmetic Test
    payload_math = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "What is 15 + 83?"}],
        "stream": False
    }
    try:
        r = requests.post(url, json=payload_math, headers=headers, timeout=10)
        print("Arithmetic test status:", r.status_code)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print("Arithmetic response:", content)
        if "98" in content:
            print("Arithmetic test: SUCCESS")
        else:
            print("Arithmetic test: FAILED")
    except Exception as e:
        print("Arithmetic test: ERROR -", e)

    # 3. Test CORS Headers
    try:
        # Check OPTIONS headers
        cors_headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        r_options = requests.options(url, headers=cors_headers, timeout=5)
        print("CORS OPTIONS status:", r_options.status_code)
        print("CORS Allow-Origin:", r_options.headers.get("access-control-allow-origin"))
        print("CORS Allow-Methods:", r_options.headers.get("access-control-allow-methods"))
    except Exception as e:
        print("CORS test: ERROR -", e)

def main():
    print("Testing local endpoint...")
    test_endpoint("http://127.0.0.1:9700/v1/chat/completions")
    
    print("\nWaiting for tunnel URL...")
    for _ in range(30):
        url = get_tunnel_url()
        if url:
            print(f"Found tunnel URL: {url}")
            test_endpoint(f"{url}/v1/chat/completions")
            break
        time.sleep(2)
    else:
        print("Tunnel URL not found in llm_tunnel.log")

if __name__ == "__main__":
    main()

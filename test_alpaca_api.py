#!/usr/bin/env python3
import os
import requests
from datetime import datetime

# Get credentials from environment
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not api_key or not secret_key:
    print("ERROR: API credentials not found in environment variables")
    exit(1)

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

print(f"Testing Alpaca API at {base_url}")
print("=" * 50)

try:
    # Test basic connectivity
    response = requests.get(f"{base_url}/v2/account", headers=headers)

    if response.status_code == 200:
        account = response.json()
        print("✓ API Connection: SUCCESS")
        print("\nAccount Details:")
        print(f"  Account Status: {account.get('status')}")
        print(f"  Account Number: {account.get('account_number')}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"  Last Updated: {account.get('last_updated_at')}")
    else:
        print(f"✗ API Error: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"✗ Connection Error: {str(e)}")

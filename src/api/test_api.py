#!/usr/bin/env python
"""
Complete API test script.
"""

import requests
import json
import time
import sys


def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    
    print("=" * 60)
    print("COMPLETE API TEST")
    print("=" * 60)
    print(f"Testing API at: {base_url}")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 2: Health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   Model loaded: {data.get('model_loaded', 'unknown')}")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 3: Model info endpoint
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   Model type: {data.get('model_type', 'unknown')}")
            print(f"   Total features: {data.get('feature_count', 'unknown')}")
            tests_passed += 1
        elif response.status_code == 503:
            print(f"   ⚠️  Model not loaded (503)")
            print(f"   Please ensure the model is trained and loaded")
        else:
            print(f"   ❌ Status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 4: Prediction endpoint
    print("\n4. Testing prediction endpoint...")
    
    # Sample data
    sample_data = {
        "limit_bal": 20000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 35,
        "pay_0": -1,
        "bill_amt1": 5000,
        "pay_amt1": 2000,
        "bill_amt2": 4500,
        "bill_amt3": 4000,
        "bill_amt4": 3500,
        "bill_amt5": 3000,
        "bill_amt6": 2500,
        "pay_amt2": 1800,
        "pay_amt3": 1600,
        "pay_amt4": 1400,
        "pay_amt5": 1200,
        "pay_amt6": 1000,
        "pay_2": -1,
        "pay_3": -1,
        "pay_4": -1,
        "pay_5": -1,
        "pay_6": -1
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   Prediction: {'Default' if result['default_prediction'] == 1 else 'No Default'}")
            print(f"   Probability: {result['default_probability']:.4f}")
            print(f"   Risk level: {result['risk_level']}")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {response.status_code}")
            print(f"   Response: {response.text}")
            tests_failed += 1
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 5: OpenAPI documentation
    print("\n5. Testing OpenAPI documentation...")
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ OpenAPI schema available")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Success rate: {tests_passed/(tests_passed+tests_failed)*100:.1f}%")
    
    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\n🎉 API is working correctly!")
        print(f"\nYou can now:")
        print(f"1. Visit API documentation: {base_url}/docs")
        print(f"2. Test predictions via the web interface")
        print(f"3. Integrate with your applications")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nTroubleshooting steps:")
        print(f"1. Ensure API is running: python run_api.py")
        print(f"2. Check if model is trained: python src/models/simple_train_fixed.py")
        print(f"3. Verify network connectivity")
    
    return tests_failed == 0


def wait_for_api(base_url="http://localhost:8000", max_retries=10):
    """Wait for API to be ready"""
    print(f"\nWaiting for API to be ready...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ API is ready!")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"   Attempt {i+1}/{max_retries}: API not ready, waiting 2 seconds...")
                time.sleep(2)
            else:
                print(f"❌ API not responding after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(2)
    
    return False


def main():
    """Main function"""
    
    # Check if API is running
    base_url = "http://localhost:8000"
    
    print("\nChecking if API is running...")
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is already running!")
        else:
            print(f"⚠️  API returned status: {response.status_code}")
            print("Please start the API first with: python run_api.py")
            return 1
    except requests.exceptions.ConnectionError:
        print("❌ API is not running")
        print("\nPlease start the API in a separate terminal:")
        print("python run_api.py")
        print("\nThen run this test again.")
        return 1
    
    # Run tests
    success = test_api_endpoints(base_url)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
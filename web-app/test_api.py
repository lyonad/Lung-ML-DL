"""
API Testing Script

Quick test script to verify API endpoints are working correctly.

Usage:
    python test_api.py
    python test_api.py --url http://localhost:5000
"""

import requests
import json
import argparse
from datetime import datetime


# Sample test data
TEST_PATIENT = {
    "age": 65,
    "gender": "M",
    "smoking": 2,
    "yellow_fingers": 2,
    "anxiety": 1,
    "peer_pressure": 1,
    "chronic_disease": 2,
    "fatigue": 2,
    "allergy": 2,
    "wheezing": 2,
    "alcohol_consuming": 1,
    "coughing": 2,
    "shortness_of_breath": 2,
    "swallowing_difficulty": 2,
    "chest_pain": 2
}


def test_health(base_url):
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Models Loaded: {data.get('models_loaded')}")
            print("✅ PASSED")
            return True
        else:
            print(f"❌ FAILED: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def test_models_info(base_url):
    """Test models info endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Models Info")
    print("="*80)
    
    try:
        response = requests.get(f"{base_url}/api/models/info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            
            models = data.get('models', {})
            print(f"\nModels:")
            for model_name, model_info in models.items():
                print(f"  - {model_name}:")
                print(f"    Accuracy: {model_info.get('accuracy')}")
                print(f"    Loaded: {model_info.get('loaded')}")
            
            dataset = data.get('dataset', {})
            print(f"\nDataset:")
            print(f"  Total Samples: {dataset.get('total_samples')}")
            print(f"  Features: {dataset.get('features')}")
            
            print("✅ PASSED")
            return True
        else:
            print(f"❌ FAILED: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def test_prediction(base_url):
    """Test prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 3: Single Prediction")
    print("="*80)
    
    print(f"Patient Data:")
    print(f"  Age: {TEST_PATIENT['age']}")
    print(f"  Gender: {TEST_PATIENT['gender']}")
    print(f"  Smoking: {'Yes' if TEST_PATIENT['smoking'] == 2 else 'No'}")
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=TEST_PATIENT,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            
            predictions = data.get('predictions', {})
            print(f"\nPredictions:")
            
            if 'regularized_ann' in predictions:
                ann = predictions['regularized_ann']
                print(f"  Regularized ANN:")
                print(f"    Prediction: {ann.get('prediction')}")
                print(f"    Confidence: {ann.get('confidence')}")
            
            if 'random_forest' in predictions:
                rf = predictions['random_forest']
                print(f"  Random Forest:")
                print(f"    Prediction: {rf.get('prediction')}")
                print(f"    Confidence: {rf.get('confidence')}")
            
            recommendation = data.get('recommendation')
            print(f"\nRecommendation:")
            print(f"  {recommendation}")
            
            print("\n✅ PASSED")
            return True
        else:
            print(f"❌ FAILED")
            error_data = response.json()
            print(f"Error: {error_data.get('error')}")
            if 'details' in error_data:
                print(f"Details: {error_data['details']}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def test_batch_prediction(base_url):
    """Test batch prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 4: Batch Prediction")
    print("="*80)
    
    # Create batch with 2 patients
    batch_data = [TEST_PATIENT, TEST_PATIENT.copy()]
    batch_data[1]['age'] = 45
    batch_data[1]['smoking'] = 1
    
    print(f"Batch Size: {len(batch_data)}")
    
    try:
        response = requests.post(
            f"{base_url}/api/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Total Processed: {data.get('total')}")
            
            results = data.get('results', [])
            print(f"\nResults:")
            for result in results:
                print(f"  Patient {result.get('index') + 1}:")
                print(f"    Success: {result.get('success')}")
            
            print("\n✅ PASSED")
            return True
        else:
            print(f"❌ FAILED: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def test_invalid_input(base_url):
    """Test API with invalid input"""
    print("\n" + "="*80)
    print("TEST 5: Invalid Input Handling")
    print("="*80)
    
    invalid_data = {
        "age": 150,  # Invalid age
        "gender": "X"  # Invalid gender
    }
    
    print(f"Sending invalid data...")
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            data = response.json()
            print(f"Error caught: {data.get('error')}")
            print("✅ PASSED (correctly rejected invalid input)")
            return True
        else:
            print(f"❌ FAILED: Should have returned 400 status code")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def run_all_tests(base_url):
    """Run all API tests"""
    print("\n" + "="*80)
    print("LUNG CANCER PREDICTION API - TEST SUITE")
    print("="*80)
    print(f"Base URL: {base_url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health),
        ("Models Info", test_models_info),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Input", test_invalid_input)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func(base_url)
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    print("="*80 + "\n")
    
    return passed == total


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test Lung Cancer Prediction API'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:5000',
        help='Base URL of the API (default: http://localhost:5000)'
    )
    
    args = parser.parse_args()
    
    # Check if server is reachable
    try:
        response = requests.get(args.url, timeout=5)
        print(f"✅ Server is reachable at {args.url}")
    except requests.exceptions.RequestException:
        print(f"❌ ERROR: Cannot reach server at {args.url}")
        print("Make sure the application is running:")
        print("  python backend/app.py")
        print("  or")
        print("  python run.py")
        return
    
    # Run tests
    success = run_all_tests(args.url)
    
    # Exit code
    exit(0 if success else 1)


if __name__ == '__main__':
    main()


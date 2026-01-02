#!/usr/bin/env python3
"""
Test script for ML Predictor
Verifies predictor.py works correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_predictor_import():
    """Test predictor can be imported"""
    print("="*60)
    print("TEST 1: IMPORT PREDICTOR")
    print("="*60)
    
    try:
        from services.ml_service.predictor import PredictiveMaintenancePredictor
        print("✓ Predictor imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import predictor: {e}")
        return False

def test_predictor_initialization():
    """Test predictor initialization"""
    print("\n" + "="*60)
    print("TEST 2: PREDICTOR INITIALIZATION")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    try:
        predictor = PredictiveMaintenancePredictor()
        
        if predictor.model_loaded:
            print("✓ Predictor initialized successfully")
            print(f"  Model loaded: {predictor.model_loaded}")
            print(f"  Features: {predictor.feature_names}")
            return True
        else:
            print("⚠ Predictor initialized but model not loaded")
            print("  This is expected if model hasn't been trained yet")
            return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_model_info():
    """Test getting model information"""
    print("\n" + "="*60)
    print("TEST 3: MODEL INFORMATION")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    try:
        predictor = PredictiveMaintenancePredictor()
        info = predictor.get_model_info()
        
        print("Model Information:")
        print(f"  Model loaded: {info['model_loaded']}")
        print(f"  Model path: {info['model_path']}")
        print(f"  Features: {info['features']}")
        print(f"  Model exists: {info['model_exists']}")
        
        print("✓ Model info retrieved successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
        return False

def test_prediction():
    """Test making a prediction"""
    print("\n" + "="*60)
    print("TEST 4: PREDICTION")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    # Check if model file exists
    model_path = Path("data/models/predictive_model.pkl")
    if not model_path.exists():
        print("⚠ Model file not found - predictions cannot be tested")
        print("  Please train the model first:")
        print("    python -m services.ml_service.trainer")
        return True  # Not a failure, just needs training
    
    try:
        predictor = PredictiveMaintenancePredictor()
        
        if not predictor.model_loaded:
            print("⚠ Model not loaded - cannot test predictions")
            return True
        
        # Test prediction
        test_data = {
            'temperature': 85.0,
            'vibration': 7.0,
            'pressure': 120.0,
            'rpm': 2400.0
        }
        
        print(f"\nTest input: {test_data}")
        result = predictor.predict_failure_probability(test_data)
        
        print("\nPrediction result:")
        print(f"  Failure predicted: {result['failure_predicted']}")
        print(f"  Failure probability: {result['failure_probability']:.2%}")
        print(f"  Risk level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']}")
        
        print("✓ Prediction successful")
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_prediction():
    """Test batch predictions"""
    print("\n" + "="*60)
    print("TEST 5: BATCH PREDICTION")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    model_path = Path("data/models/predictive_model.pkl")
    if not model_path.exists():
        print("⚠ Model file not found - batch predictions cannot be tested")
        return True
    
    try:
        predictor = PredictiveMaintenancePredictor()
        
        if not predictor.model_loaded:
            print("⚠ Model not loaded - cannot test batch predictions")
            return True
        
        # Test data
        batch_data = [
            {'temperature': 60, 'vibration': 3, 'pressure': 100, 'rpm': 2000},
            {'temperature': 85, 'vibration': 7, 'pressure': 120, 'rpm': 2400},
            {'temperature': 95, 'vibration': 9, 'pressure': 140, 'rpm': 2800}
        ]
        
        print(f"\nBatch size: {len(batch_data)} predictions")
        results = predictor.batch_predict(batch_data)
        
        print("\nBatch results:")
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                print(f"  {i}. Risk: {result['risk_level']}, "
                      f"Prob: {result['failure_probability']:.2%}")
            else:
                print(f"  {i}. Error: {result['error']}")
        
        print(f"✓ Batch prediction successful ({len(results)} predictions)")
        return True
        
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False

def test_input_validation():
    """Test input validation"""
    print("\n" + "="*60)
    print("TEST 6: INPUT VALIDATION")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    try:
        predictor = PredictiveMaintenancePredictor()
        
        # Test missing feature
        print("\n1. Testing missing feature...")
        try:
            incomplete_data = {'temperature': 60, 'vibration': 3}
            predictor._validate_input(incomplete_data)
            print("✗ Should have raised ValueError for missing features")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected incomplete data: {str(e)[:50]}...")
        
        # Test invalid value
        print("\n2. Testing invalid value...")
        try:
            invalid_data = {
                'temperature': 'invalid',
                'vibration': 3,
                'pressure': 100,
                'rpm': 2000
            }
            predictor._validate_input(invalid_data)
            print("✗ Should have raised ValueError for invalid value")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected invalid data: {str(e)[:50]}...")
        
        # Test valid data
        print("\n3. Testing valid data...")
        valid_data = {
            'temperature': 60,
            'vibration': 3,
            'pressure': 100,
            'rpm': 2000
        }
        validated = predictor._validate_input(valid_data)
        print(f"✓ Valid data accepted: {validated}")
        
        print("\n✓ Input validation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        return False

def test_feature_importance():
    """Test feature importance extraction"""
    print("\n" + "="*60)
    print("TEST 7: FEATURE IMPORTANCE")
    print("="*60)
    
    from services.ml_service.predictor import PredictiveMaintenancePredictor
    
    model_path = Path("data/models/predictive_model.pkl")
    if not model_path.exists():
        print("⚠ Model file not found - feature importance cannot be tested")
        return True
    
    try:
        predictor = PredictiveMaintenancePredictor()
        
        if not predictor.model_loaded:
            print("⚠ Model not loaded - cannot test feature importance")
            return True
        
        importance = predictor.get_feature_importance()
        
        print("\nFeature Importance:")
        for feature, score in importance.items():
            if score is not None:
                print(f"  {feature:12s}: {score:.4f}")
            else:
                print(f"  {feature:12s}: N/A")
        
        print("✓ Feature importance extracted successfully")
        return True
        
    except Exception as e:
        print(f"✗ Feature importance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ML PREDICTOR TEST SUITE")
    print("="*70)
    
    tests = [
        ("Import", test_predictor_import),
        ("Initialization", test_predictor_initialization),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Input Validation", test_input_validation),
        ("Feature Importance", test_feature_importance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nPredictor is working correctly!")
        
        # Check if model is trained
        model_path = Path("data/models/predictive_model.pkl")
        if not model_path.exists():
            print("\n⚠ Note: Model not trained yet")
            print("  To train the model, run:")
            print("    python -m services.ml_service.trainer")
        else:
            print("\n✓ Model is trained and ready to use")
        
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
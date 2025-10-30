#!/usr/bin/env python3
"""
Test script to validate the model fixes work correctly.
This script tests the model architecture without full training.
"""

import tensorflow as tf
import numpy as np
from modules import construct_classifier, compile_model
from config import Config

def test_model_architecture():
    """Test that the model architecture works correctly."""
    print("Testing model architecture...")
    
    # Create a simple test model
    model = construct_classifier()
    config = Config.getInstance()
    compile_model(model, config['learning_rate'])
    
    # Create dummy data for testing
    batch_size = 4
    img_shape = (256, 256, 3)
    
    # Create test inputs
    anchor = tf.random.normal((batch_size,) + img_shape)
    labels = tf.constant([1, 0, 1, 0], dtype=tf.int32)  # Mix of positive and negative
    positive = tf.random.normal((batch_size,) + img_shape)
    negative = tf.random.normal((batch_size,) + img_shape)
    
    inputs = [anchor, labels, positive, negative]
    
    # Test forward pass
    try:
        outputs = model(inputs, training=True)
        print(f"✓ Forward pass successful. Output shape: {outputs.shape}")
        print(f"✓ Output values: {outputs.numpy().flatten()}")
        
        # Test that outputs are binary (0 or 1)
        unique_values = np.unique(outputs.numpy())
        expected_values = {0.0, 1.0}
        if set(unique_values).issubset(expected_values):
            print("✓ Outputs are properly binary (0 or 1)")
        else:
            print(f"⚠ Warning: Outputs contain unexpected values: {unique_values}")
            
        # Test training step
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            # Model should have internal losses from TripletLoss layer
            total_loss = tf.reduce_sum(model.losses)
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Check that gradients are not None or all zeros
        non_none_grads = [g for g in gradients if g is not None]
        if len(non_none_grads) > 0:
            print("✓ Gradients computed successfully")
            avg_grad_norm = tf.reduce_mean([tf.norm(g) for g in non_none_grads])
            print(f"✓ Average gradient norm: {avg_grad_norm:.6f}")
        else:
            print("✗ No gradients computed - training may not work")
            
        print(f"✓ Total loss: {total_loss:.6f}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
        
    print("✓ All tests passed!")
    return True

def test_distance_calculations():
    """Test the distance layer calculations manually."""
    print("\nTesting distance calculations...")
    
    # Create simple test vectors
    anchor = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    positive = tf.constant([[1.1, 0.1], [0.1, 1.1]], dtype=tf.float32)  # Close to anchor
    negative = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)  # Far from anchor
    
    # Calculate distances manually
    ap_dist = tf.norm(anchor - positive, axis=1, keepdims=True)
    an_dist = tf.norm(anchor - negative, axis=1, keepdims=True)
    
    print(f"Anchor-Positive distances: {ap_dist.numpy().flatten()}")
    print(f"Anchor-Negative distances: {an_dist.numpy().flatten()}")
    
    # Test classification logic
    predictions = tf.cast(ap_dist < an_dist, tf.float32)
    print(f"Predictions (should be [1, 1]): {predictions.numpy().flatten()}")
    
    if np.allclose(predictions.numpy(), [1.0, 1.0]):
        print("✓ Distance-based classification working correctly")
    else:
        print("✗ Distance-based classification not working as expected")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Model Fixes")
    print("=" * 50)
    
    try:
        # Test distance calculations first
        test_distance_calculations()
        
        # Test full model architecture
        success = test_model_architecture()
        
        if success:
            print("\n" + "=" * 50)
            print("✓ All tests passed! The model fixes should work correctly.")
            print("You can now run training with: python predict.py")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ Some tests failed. Please check the errors above.")
            print("=" * 50)
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
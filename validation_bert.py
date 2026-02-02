import sys

def test_bert_compatibility():
    print("--- Starting BERT (NAACL 2019) Compatibility Check ---")
    
    try:
        import tensorflow as tf
        print(f"--> Detected TensorFlow version: {tf.__version__}")

        # 1. Check for tf.flags (Removed in TF 2.0)
        print("--> Accessing tf.flags...")
        flags = tf.flags
        FLAGS = flags.FLAGS
        print("    [✓] tf.flags found.")

        # 2. Check for tf.contrib (Removed in TF 2.0)
        print("--> Accessing tf.contrib.layers...")
        # BERT uses this for layer normalization and initializers
        contrib = tf.contrib.layers
        print("    [✓] tf.contrib found.")

        # 3. Check for Static Graph APIs (Disabled by default in TF 2.0)
        print("--> Initializing Global Variables Initializer...")
        init = tf.global_variables_initializer()
        print("    [✓] Static Graph API found.")

        print("--- SMOKE TEST PASSED ---")

    except AttributeError as ae:
        print(f"CRITICAL VALIDATION FAILURE: {str(ae)}")
        print("HINT: This code requires TensorFlow 1.x. TF 2.x is incompatible.")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_bert_compatibility()
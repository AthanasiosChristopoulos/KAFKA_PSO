import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   

def export_mobilenetv2_base(save_path="pretrained_model/mobilenetv2_base_32x32.h5"):
    # Feature extractor only (no classifier head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,      # <- important
        weights="imagenet"
    )

    # Freeze it (feature extractor mode)
    base_model.trainable = False

    _ = base_model(tf.zeros([1, 32, 32, 3]), training=False)

    print("\n=== Base model summary ===")
    base_model.summary()

    base_model.save(save_path)
    print(f"\nSaved base model to: {save_path}")

    print("\nLast 20 layer names (for DL4J setFeatureExtractor / removing vertices):")
    for layer in base_model.layers[-20:]:
        print("  ", layer.name)

# ============================================================================================

def export_mobilenetv3small_base(save_path="pretrained_model/mobilenetv3small_32x32.h5"):

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_model.summary()

    base_model.save(save_path)

# ============================================================================================

if __name__ == "__main__":
    
    export_mobilenetv2_base()
    # export_mobilenetv3small_base()

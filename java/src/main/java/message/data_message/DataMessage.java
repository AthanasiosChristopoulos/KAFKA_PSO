
    package message.data_message;


    public class DataMessage {
        public int sampleIndex;
        public float[] features;
        public int label;

        public DataMessage(int sampleIndex, float[] features, int label) {
            this.sampleIndex = sampleIndex;
            this.features = features;
            this.label = label;
        }

        // DataMessage size estimation for ONE CIFAR-10 image flattened to float[] features
        // CIFAR image shape = (32, 32, 3)
        // flattened length = 32 * 32 * 3 = 3072 floats

        // int sampleIndex = 4 bytes
        // int label       = 4 bytes

        // float[] features (length = 3072)
        //   each float = 4 bytes
        //   3072 * 4 = 12288 bytes
        //   array overhead ≈ 16 bytes
        //   total features ≈ 12304 bytes

        // Total ≈ 4 + 12304 + 4 = 12312 bytes

        // ≈ 12.0 KB per CIFAR record

        // ======================================================================

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("sampleIndex = ").append(sampleIndex);
            sb.append(", label = ").append(label);

            sb.append(", featuresSample = [");

            int n = Math.min(5, features.length);
            for (int i = 0; i < n; i++) {
                sb.append(String.format("%.5f", features[i]));
                if (i < n - 1) sb.append(", ");
            }
            
            sb.append(", ...]}");
            return sb.toString();
        }

        // ======================================================================

        public String toStringFull() {
            StringBuilder sb = new StringBuilder();
            sb.append("sampleIndex=").append(sampleIndex);
            sb.append(", label=").append(label);

            sb.append(", featuresSample=[");

            for (int i = 0; i < features.length; i++) {
                sb.append(String.format("%.5f", features[i]));
                if (i < features.length - 1) sb.append(", ");
            }
            
            sb.append(", ...]}");
            return sb.toString();
        }
    }

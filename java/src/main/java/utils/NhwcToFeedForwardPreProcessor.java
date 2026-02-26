package utils;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;              // ✅ correct for M2.1
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;

public class NhwcToFeedForwardPreProcessor implements InputPreProcessor {
    private final int h, w, c;

    public NhwcToFeedForwardPreProcessor(int h, int w, int c) {
        this.h = h; this.w = w; this.c = c;
    }

    @Override
    public INDArray preProcess(INDArray input, int minibatchSize, LayerWorkspaceMgr workspaceMgr) {
        // expects NHWC: [N,H,W,C] (incoming is nhwc (coming from python) needs to be flattend)
        long[] s = input.shape();
        if (input.rank() != 4 || s[1] != h || s[2] != w || s[3] != c) {
            throw new IllegalStateException(
                "Expected NHWC ["+minibatchSize+","+h+","+w+","+c+"] but got " + java.util.Arrays.toString(s)
            );
        }
        // flatten -> [N, H*W*C]
        return input.reshape('c', minibatchSize, h * w * c);    // just a flattend array
        // There are two common orderings:
        // 'c' = C order (row-major, like NumPy default): the last dimension changes fastest
        // 'f' = Fortran order (column-major): the first dimension changes fastest

    }

    @Override
    public INDArray backprop(INDArray output, int minibatchSize, LayerWorkspaceMgr workspaceMgr) {
        // [N, H*W*C] -> [N,H,W,C]
        return output.reshape('c', minibatchSize, h, w, c);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        return InputType.feedForward(h * w * c);
    }
    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        // No sequence/time dimension here; mask unchanged
        return Pair.of(maskArray, currentMaskState);
    }

    @Override
    public NhwcToFeedForwardPreProcessor clone() {
        return new NhwcToFeedForwardPreProcessor(h, w, c);
    }
}
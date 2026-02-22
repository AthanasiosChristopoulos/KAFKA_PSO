package utils;
import org.bytedeco.javacpp.SizeTPointer;
import static org.bytedeco.cuda.global.cudart.*;

public class GpuMem {
    public static void log(String tag){
        SizeTPointer free = new SizeTPointer(1);
        SizeTPointer total = new SizeTPointer(1);

        int rc = cudaMemGetInfo(free, total);
        if(rc != 0){
            System.out.println(tag + " cudaMemGetInfo failed rc=" + rc);
            return;
        }

        long freeB = free.get();
        long totalB = total.get();
        long usedB = totalB - freeB;

        System.out.printf(
            "%s GPU used=%.2f MB free=%.2f MB total=%.2f MB%n",
            tag,
            usedB / 1024.0 / 1024.0,
            freeB / 1024.0 / 1024.0,
            totalB / 1024.0 / 1024.0
        );

        free.close();
        total.close();
    }
}
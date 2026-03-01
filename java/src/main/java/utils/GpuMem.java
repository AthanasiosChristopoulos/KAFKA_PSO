package utils;
import org.bytedeco.javacpp.SizeTPointer;
import static org.bytedeco.cuda.global.cudart.*;

public class GpuMem {


    public static void log(String tag){

        if(1 == 1) {
            return;
        }

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

    // =================================================================================
    
    public static double usedMb() {
        SizeTPointer free = new SizeTPointer(1);
        SizeTPointer total = new SizeTPointer(1);

        int rc = cudaMemGetInfo(free, total);
        if (rc != 0) return -1;

        long freeB = free.get();
        long totalB = total.get();
        long usedB = totalB - freeB;

        free.close();
        total.close();
        return usedB / 1024.0 / 1024.0;
    }

    // =================================================================================

    public static double freeMb() {
        SizeTPointer free = new SizeTPointer(1);
        SizeTPointer total = new SizeTPointer(1);

        int rc = cudaMemGetInfo(free, total);
        if (rc != 0) {
            System.out.println("cudaMemGetInfo failed rc=" + rc);
            free.close();
            total.close();
            return -1;
        }

        long freeB = free.get();

        free.close();
        total.close();

        return freeB / 1024.0 / 1024.0;
    }

    // =================================================================================

    public static double totalMb() {
        SizeTPointer free = new SizeTPointer(1);
        SizeTPointer total = new SizeTPointer(1);

        int rc = cudaMemGetInfo(free, total);
        if (rc != 0) {
            free.close();
            total.close();
            return -1;
        }

        long totalB = total.get();

        free.close();
        total.close();

        return totalB / 1024.0 / 1024.0;
    }

    // =================================================================================
    public static double usedPercent() {
        double used = usedMb();
        double total = totalMb();

        if (used < 0 || total <= 0) return -1;

        return used / total;   // value in range [0,1]
    }
}
# fadv.py DIR [SECONDS] : drop DIR's *.safetensors from the page cache every SECONDS while a load runs (no root needed)
import glob, os, sys, time
d, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 15
end = time.time() + float(os.environ.get("FADV_FOR_S", "1800"))
while time.time() < end:
    for f in glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True):
        try:
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass
    time.sleep(period)

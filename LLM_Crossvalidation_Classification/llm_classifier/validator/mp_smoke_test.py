from multiprocessing import Process
import time

def worker(i):
    import onnxruntime as ort
    print(f"Worker {i} providers:", ort.get_available_providers())
    time.sleep(5)

if __name__ == "__main__":
    ps = [Process(target=worker, args=(i,)) for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()

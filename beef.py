import pyopencl as cl
import itertools
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("a", type=int) 
parser.add_argument("outputs", nargs="+", type=float)
parser.add_argument("--cl-platform", default=0, type=int)
parser.add_argument("--cl-device", default=0, type=int)
parser.add_argument("--max-results", default=16, type=int)
parser.add_argument("--max-skip", default=32, type=int)
parser.add_argument("--dump-length", default=64, type=int)
args = parser.parse_args()

platform = cl.get_platforms()[args.cl_platform]
device = platform.get_devices()[args.cl_device]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

kernel_src=open("generic.cl").read()

program = cl.Program(ctx, kernel_src).build()

max_results = args.max_results 
mem_flags = cl.mem_flags

queue = cl.CommandQueue(ctx)

def normalize(n):
    return float(int(n * (1<<32))) / (1<<32)

def toint(n):
    return int(n / np.float64(2.3283064365386962890625e-10))

uint32_size = 4

full_outputs = map(toint, map(normalize, args.outputs))

def core_search(partial_outputs, a):
        search_len_np = np.array(args.max_skip, dtype=np.uint32)
        result_count_cl = cl.Buffer(ctx, mem_flags.READ_WRITE, uint32_size)
        result_cl = cl.Buffer(ctx, mem_flags.WRITE_ONLY, max_results*uint32_size) 
        cl.enqueue_fill_buffer(queue, result_count_cl, "\x00", 0, uint32_size)
        result_max_np = np.array(max_results, dtype=np.uint32)
        a_np = np.array(a, dtype=np.uint32)
        outputs_np = np.array(partial_outputs, dtype=np.uint32)
        outputs_cl = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=outputs_np) 
        outputs_len_np = np.array(len(outputs_np), dtype=np.uint32)

        program.node_newer_rng(queue, [1<<16], None, search_len_np, result_max_np, a_np, outputs_cl, outputs_len_np, result_count_cl, result_cl)

        return result_count_cl, result_cl

print "Starting search for r0"
(result_count_r0_cl, result_r0_cl) = core_search(map(lambda x: ((x >> 16)&0xFFFF), full_outputs), args.a)
print "Starting search for r1"
(result_count_r1_cl, result_r1_cl) = core_search(map(lambda x: ((x >> 16)&0xFFFF), full_outputs), args.a)

result_count_r0_np = np.array(0, dtype=np.uint32)
result_r0_np = np.zeros(max_results, np.uint32)
result_count_r1_np = np.array(0, dtype=np.uint32)
result_r1_np = np.zeros(max_results, np.uint32)

cl.enqueue_copy(queue, result_count_r0_np, result_count_r0_cl)
cl.enqueue_copy(queue, result_r0_np, result_r0_cl)
cl.enqueue_copy(queue, result_count_r1_np, result_count_r1_cl)
cl.enqueue_copy(queue, result_r1_np, result_r1_cl)

np.set_printoptions(precision=32)

print "Found", result_count_r0_np, "results for r0."
print "Found", result_count_r1_np, "results for r1."
results = itertools.product(
        result_r0_np[0:result_count_r0_np],
        result_r1_np[0:result_count_r1_np]
        )
for j, k in results: 
    print "\tState: (" + str(j) + ", "+ str(k) + ")"
    for i in range(args.dump_length):
        j = (args.a * (j & 0xFFFF)) + (j >> 16)
        k = (36969 * (k & 0xFFFF)) + (k >> 16)
        x = np.uint32((j << 16) + (k & 0xFFFF))
        pred = x * np.float64("2.3283064365386962890625e-10")
        print "\t\t" + ("%.20f" % normalize(pred))

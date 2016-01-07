import pyopencl as cl
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("kernel", help="The kernel to use; 18030.cl or 18273.cl")
parser.add_argument("outputs", nargs="+", type=float)
parser.add_argument("--cl-platform", default=0)
parser.add_argument("--cl-device", default=0)
parser.add_argument("--max-results", default=16)
parser.add_argument("--max-skip", default=32)
args = parser.parse_args()

platform = cl.get_platforms()[args.cl_platform]
device = platform.get_devices()[args.cl_device]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

kernel_src = open(args.kernel).read()
program = cl.Program(ctx, kernel_src).build()

max_results = args.max_results 
mem_flags = cl.mem_flags
result_np = np.zeros(max_results, np.uint32)
result_cl = cl.Buffer(ctx, mem_flags.WRITE_ONLY, result_np.nbytes);

queue = cl.CommandQueue(ctx)
search_len_np =  np.array(args.max_skip, dtype=np.uint32)
result_count_np =  np.array(0, dtype=np.uint32)
result_count_cl = cl.Buffer(ctx, mem_flags.READ_WRITE, result_count_np.nbytes);
result_max_np =  np.array(max_results, dtype=np.uint32)

full_outputs = args.outputs 
state_sample = int(full_outputs[0] / 2.3283064365386962890625e-10)

lower_state_np = np.array([state_sample>>16, state_sample&0xFFFF], dtype=np.uint32)

outputs_np = np.array(full_outputs[1:], dtype=np.float64)
outputs_cl =  cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=outputs_np) 

outputs_len_np = np.array(len(outputs_np), dtype=np.uint32)
print "Starting search..."
program.node_newer_rng(queue, [(1<<32) - 1], None, lower_state_np, search_len_np, result_max_np, outputs_cl, outputs_len_np, result_count_cl, result_cl)

cl.enqueue_copy(queue, result_count_np, result_count_cl)
cl.enqueue_copy(queue, result_np, result_cl)

print "Found", result_count_np, "results."
print "States:"
for q in result_np[0:result_count_np]:
    j = lower_state_np[0] | ((q & 0xFFFF)<<16)
    k = lower_state_np[1] | (q & 0xFFFF0000)

    print "\t("+j + ", "+ k + ")"
print "Run with --make-predictions (state_first, state_last) to predict future outputs."

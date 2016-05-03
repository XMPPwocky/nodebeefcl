# nodebeefcl
Recovers V8 Math.random seed from outputs, even when they are not consecutive.

# Usage

````python beef.py 18273 0.12345678 0.42424242 0.86737282````

Replace 18273 with 18030 to handle different versions of V8/Node.

# Dependencies

Python 2.x, numpy, PyOpenCL.


# Examples
## Running nodebeefcl on Debian and Intel i915 GPU
### Install the dependencies

```Bash

$ sudo apt-get install python-pyopencl beignet-opencl-icd

#
# You'll need to disable the i915 hang check otherwise `dmesg` will be filled with
# this:
#
# [143867.599237] [drm] GPU HANG: ecode 7:0:0x8fd8ffff, in python [24620], reason: Ring hung, action: reset
# [143867.601402] drm/i915: Resetting chip after gpu hang
# [143889.586677] [drm] stuck on render ring
#

$ sudo bash -c 'echo N > /sys/module/i915/parameters/enable_hangcheck'

#

$ python beef.py 18273 0.8102033962495625 0.8550206781364977
Starting search...
Found 13 results.
    State: (1234620265, 179273058)
            0.99755238764919340611
            0.33525288174860179424

# Victory !
```

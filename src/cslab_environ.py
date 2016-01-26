import sys
import os

if os.path.exists('/u/mren'):
    sys.path.insert(
        0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')
    sys.path.insert(
        0, '/u/mren/code/img-count/third_party/tensorflow/_python_build/')

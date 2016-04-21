# CUDA-longest-prefix-match
Longest prefix match for IPv4 implemented using CUDA technology.

graphical processor: 2.1
CUDA version: 7.5

The program creates random IPs and Masks then assigns proper IPs to Masks if not found it falls back to default 0.0.0.0
using a 255'ary tree of height 4 (each node has <0,255> children)


"""DSV4 Attention kernels — public integration API.

The live inference path uses dsv4.kernels.attention.production directly.
See production.py for the dsv4_attention function used by single_shot_inference.py.
"""
from dsv4.kernels.attention.production import dsv4_attention
from dsv4.kernels.attention.production import dsv4_attention_mixed_fp8_decode

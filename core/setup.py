from setuptools import setup, Extension

setup(
    name="matmul",
    ext_modules=[
        Extension("matmul", ["mat_mul_module.c"])
    ]
)
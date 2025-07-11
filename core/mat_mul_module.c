#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "mat_mul.c"  // Or declare:wthe function prototype if you compile separately

static PyObject* py_matmul_forward_cpu(PyObject* self, PyObject* args) {
    Py_buffer inp_buf, weight_buf, bias_buf, out_buf;
    int B, T, C, OC;
    PyObject* bias_obj;

    if (!PyArg_ParseTuple(args, "y*y*y*OiOiO", &inp_buf, &weight_buf, &bias_obj, &B, &T, &C, &OC, &out_buf))
        return NULL;

    float* inp = (float*)inp_buf.buf;
    float* weight = (float*)weight_buf.buf;
    float* bias = NULL;
    if (bias_obj != Py_None) {
        if (PyObject_GetBuffer(bias_obj, &bias_buf, PyBUF_SIMPLE) < 0) return NULL;
        bias = (float*)bias_buf.buf;
    }
    float* out = (float*)out_buf.buf;

    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    if (bias) PyBuffer_Release(&bias_buf);
    PyBuffer_Release(&inp_buf);
    PyBuffer_Release(&weight_buf);
    PyBuffer_Release(&out_buf);

    Py_RETURN_NONE;
}

static PyMethodDef MatMulMethods[] = {
    {"matmul_forward_cpu", py_matmul_forward_cpu, METH_VARARGS, "Matrix multiplication with optional bias."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matmulmodule = {
    PyModuleDef_HEAD_INIT,
    "matmul",   // name of module
    NULL,       // module documentation
    -1,
    MatMulMethods
};

PyMODINIT_FUNC PyInit_matmul(void) {
    return PyModule_Create(&matmulmodule);
}
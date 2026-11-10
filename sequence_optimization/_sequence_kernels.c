#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* Inputs must already be normalized; validate them again at the native boundary.
   Return sequence-major, position-major, alphabet-major order, excluding identity. */
static PyObject *single_mutants(PyObject *self, PyObject *args) {
    PyObject *input, *alphabet, *positions, *seqs = NULL, *pos = NULL, *out = NULL;
    if (!PyArg_ParseTuple(args, "OUO", &input, &alphabet, &positions)) return NULL;
    if (!PyUnicode_IS_ASCII(alphabet) || PyUnicode_GET_LENGTH(alphabet) < 2) {
        PyErr_SetString(PyExc_ValueError, "alphabet must contain at least two unique ASCII symbols"); return NULL;
    }
    const char *letters = PyUnicode_AsUTF8(alphabet);
    Py_ssize_t nletters = PyUnicode_GET_LENGTH(alphabet);
    for (Py_ssize_t a = 0; a < nletters; ++a)
        for (Py_ssize_t b = 0; b < a; ++b)
            if (letters[a] == letters[b]) {
                PyErr_SetString(PyExc_ValueError, "alphabet symbols must be unique"); return NULL;
            }
    seqs = PySequence_Fast(input, "sequences must be iterable");
    if (!seqs) goto fail;
    pos = PySequence_Fast(positions, "positions must be iterable");
    if (!pos) goto fail;
    out = PyList_New(0);
    if (!out) goto fail;
    Py_ssize_t expected = -1;
    for (Py_ssize_t row = 0; row < PySequence_Fast_GET_SIZE(seqs); ++row) {
        PyObject *s = PySequence_Fast_GET_ITEM(seqs, row);
        if (!PyUnicode_Check(s) || !PyUnicode_IS_ASCII(s)) {
            PyErr_SetString(PyExc_ValueError, "sequences must be ASCII strings"); goto fail;
        }
        Py_ssize_t length = PyUnicode_GET_LENGTH(s);
        if (expected >= 0 && length != expected) {
            PyErr_SetString(PyExc_ValueError, "sequences must have equal length"); goto fail;
        }
        expected = length;
        const char *original = PyUnicode_AsUTF8(s);
        for (Py_ssize_t j = 0; j < length; ++j) {
            if (!memchr(letters, original[j], nletters)) {
                PyErr_SetString(PyExc_ValueError, "sequence symbol outside alphabet"); goto fail;
            }
        }
        char *buffer = PyMem_Malloc(length + 1);
        if (!buffer) { PyErr_NoMemory(); goto fail; }
        memcpy(buffer, original, length + 1);
        for (Py_ssize_t p = 0; p < PySequence_Fast_GET_SIZE(pos); ++p) {
            Py_ssize_t index = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(pos, p));
            if (PyErr_Occurred() || index < 0 || index >= length) {
                PyMem_Free(buffer);
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "position outside sequence");
                goto fail;
            }
            for (Py_ssize_t a = 0; a < nletters; ++a) {
                if (letters[a] == original[index]) continue;
                buffer[index] = letters[a];
                PyObject *mutant = PyUnicode_FromStringAndSize(buffer, length);
                if (!mutant) { PyMem_Free(buffer); goto fail; }
                int status = PyList_Append(out, mutant);
                Py_DECREF(mutant);
                if (status < 0) { PyMem_Free(buffer); goto fail; }
            }
            buffer[index] = original[index];
        }
        PyMem_Free(buffer);
    }
    Py_DECREF(seqs); Py_DECREF(pos);
    return out;
fail:
    Py_XDECREF(seqs); Py_XDECREF(pos); Py_XDECREF(out);
    return NULL;
}
static PyMethodDef methods[] = {
    {"single_mutants", single_mutants, METH_VARARGS, "Construct validated ASCII single substitutions."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_sequence_kernels", NULL, -1, methods};
PyMODINIT_FUNC PyInit__sequence_kernels(void) { return PyModule_Create(&module); }

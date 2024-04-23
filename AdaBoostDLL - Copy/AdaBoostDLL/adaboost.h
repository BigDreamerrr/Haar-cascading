#pragma once
#include<Python.h>
#include<arrayobject.h>
#include<Windows.h>
#include<stdbool.h>

#define CATEGORY 0
#define CONTINOUS 1

#define LEFT_RIGHT 0
#define ABOVE_BELOW 1
#define DIAGONAL 2
#define LINE 3

typedef struct Stump {
    int feature_index;
    int type;
    double value;
    int* label;
} Stump;

typedef struct AdaBoost {
    int n_classifiers;
    Stump** classifiers;
    double* weights;
    int* types;
    int step;
} AdaBoost;

__declspec(dllexport) PyArrayObject* Ada_predict2(AdaBoost* model, PyObject* X);
__declspec(dllexport) void Ada_fit(AdaBoost* model, PyObject* X, PyObject* Y);
__declspec(dllexport) AdaBoost* make_adaboost(int types_length, int* types, int n_bins, int n_classifiers);
__declspec(dllexport) void clean_ada(AdaBoost* model);
__declspec(dllexport) void* Adaboost_init();
__declspec(dllexport) PyObject* py_haar(PyObject* img, int win_size, int feature_extractor, int stride);
__declspec(dllexport) void save_adaboost(AdaBoost* model, char* path, int path_length, char* model_name);
__declspec(dllexport) AdaBoost* load_adaboost(char* path, int path_length);
__declspec(dllexport) PyObject* build_integral(int height, int width, PyObject* py_img);
__declspec(dllexport) void fill_fet_vec(
    int win_size,
    int stride,
    int block_size,
    int shape_1,
    int* indexes,
    PyObject* full_fets,
    double** integral,
    int indexes_size);
//__declspec(dllexport) double get_haar_fet_at(double** integral, int win_size, int feature_extractor, int stride, int x, int y);
__declspec(dllexport) double** create_fast_integral(PyObject* integral);
__declspec(dllexport) void delete_fast_integral(PyObject* integral, double** c_integral);
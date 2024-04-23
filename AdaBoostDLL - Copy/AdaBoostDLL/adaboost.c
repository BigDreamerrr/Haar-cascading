#include"adaboost.h"

void* Adaboost_init() {
    import_array();

    return NULL;
}

void Write_npy(PyObject* arr, char* path) {
    PyRun_SimpleString("import sys;\nsys.path.append(r\"D:\\Extensions\")\n");

    PyObject* module = PyImport_ImportModule("read");
    PyObject* func = PyObject_GetAttrString(module, "save_npy");

    PyObject* args = PyTuple_New(2);
    PyObject* py_path = PyUnicode_FromString(path);
    PyTuple_SetItem(args, 0, arr);
    PyTuple_SetItem(args, 1, py_path);

    PyObject_CallObject(func, args);
}

PyObject* Read_npy(char* path) {
    PyRun_SimpleString("import sys;\nsys.path.append(r\"D:\\Extensions\")\n");

    PyObject* module = PyImport_ImportModule("read");
    PyObject* func = PyObject_GetAttrString(module, "read_npy");

    PyObject* args = PyTuple_New(1);
    PyObject* py_path = PyUnicode_FromString(path);
    PyTuple_SetItem(args, 0, py_path);

    return PyObject_CallObject(func, args);
}

void init_stump(
    Stump* stump,
    int feature_index,
    int type,
    double** X,
    double* Y,
    int N,
    double value) {

    stump->feature_index = feature_index;
    stump->type = type;
    stump->value = value;
    stump->label = (int*)malloc(2 * sizeof(int));
    stump->label[0] = -1;
    stump->label[1] = -1;

    int group_cnt[2][2] = {
        {0, 0},
        {0, 0}
    };

    for (int i = 0; i < N; i++) {
        int group_index = -1;
        int class_index = (Y[i] == -1 ? 0 : 1);
        if (stump->type == CATEGORY) {
            group_index = (X[i][stump->feature_index] == stump->value ? 0 : 1);
        }
        else {
            group_index = (X[i][stump->feature_index] < stump->value ? 0 : 1);
        }
        group_cnt[group_index][class_index] += 1;
    }

    if (group_cnt[0][0] < group_cnt[0][1]) {
        stump->label[0] = 1;
    }
    if (group_cnt[1][0] < group_cnt[1][1]) {
        stump->label[1] = 1;
    }
}

void free_stump(Stump* stump) {
    if (stump == NULL) {
        return;
    }

    free(stump->label);
    free(stump);
}

void predict(Stump* stump, double** X, double* Y, int N) {
    for (int i = 0; i < N; i++) {
        int group_index = -1;
        if (stump->type == CATEGORY) {
            group_index = (X[i][stump->feature_index] == stump->value ? 0 : 1);
        }
        else {
            group_index = (X[i][stump->feature_index] < stump->value ? 0 : 1);
        }

        Y[i] = stump->label[group_index];
    }
}

AdaBoost* make_adaboost(int types_length, int* types, int n_bins, int n_classifiers) {
    AdaBoost* model = malloc(sizeof(AdaBoost));
    model->n_classifiers = n_classifiers;
    model->step = n_bins;

    model->classifiers = malloc(n_classifiers * sizeof(Stump*));
    memset(model->classifiers, 0, n_classifiers * sizeof(Stump*));

    model->weights = malloc(n_classifiers * sizeof(double));

    model->types = malloc(types_length * sizeof(int));
    memcpy(model->types, types, types_length * sizeof(int));

    return model;
}

void clean_ada(AdaBoost* model) {
    free(model->weights);
    Stump** freed_classifiers = malloc(sizeof(Stump*) * model->n_classifiers);
    int freed_sz = 0;

    for (int i = 0; i < model->n_classifiers; i++) {
        if (model->classifiers[i] == NULL) {
            break;
        }

        bool freed = false;
        for (int j = 0; j < freed_sz; j++) {
            if (freed_classifiers[j] == model->classifiers[i]) {
                freed = true;
                break;
            }
        }

        if (!freed) {
            freed_classifiers[freed_sz] = model->classifiers[i];
            freed_sz += 1;
            free_stump(model->classifiers[i]);
        }
    }

    free(freed_classifiers);

    free(model->types);
    free(model->classifiers);
    free(model);
}

void ada_predict(AdaBoost* self, double** X, double* Y_pred, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        Y_pred[i] = 0.0;
    }

    double* pred = malloc(n_samples * sizeof(double));
    for (int i = 0; i < self->n_classifiers; i++) {
        if (self->classifiers[i] == NULL) {
            break;
        }

        predict(self->classifiers[i], X, pred, n_samples);
        for (int j = 0; j < n_samples; j++) {
            Y_pred[j] += pred[j] * self->weights[i];
        }
    }

    free(pred);
}

double error(AdaBoost* self, double** X, double* Y, int n_samples, Stump* last_classifier, double* last_model_preds) {
    double error = 0.0;

    double* last_pred = malloc(n_samples * sizeof(double));
    predict(last_classifier, X, last_pred, n_samples);

    for (int i = 0; i < n_samples; i++) {
        if (last_pred[i] == Y[i]) {
            continue;
        }
        error += exp(-last_model_preds[i] * Y[i]);
    }

    free(last_pred);
    return error;
}

double loss(AdaBoost* self, double** X, double* Y, int n_samples) {
    double loss = 0.0;
    double* Y_pred = malloc(n_samples * sizeof(double));

    ada_predict(self, X, Y_pred, n_samples);

    for (int i = 0; i < n_samples; i++) {
        loss += exp(-Y_pred[i] * Y[i]);
    }

    free(Y_pred);
    return loss;
}

void fit(AdaBoost* self, double** X, double* Y, int n_samples, int n_features) {
    double* from_values = malloc(sizeof(double) * n_features);
    double* to_values = malloc(sizeof(double) * n_features);

    for (int j = 0; j < n_features; j++) {
        double from_value = INFINITY;
        double to_value = -INFINITY;

        for (int k = 0; k < n_samples; k++) {
            double value = X[k][j];
            if (value < from_value) {
                from_value = value;
            }
            if (value > to_value) {
                to_value = value;
            }
        }

        from_values[j] = from_value;
        to_values[j] = to_value;
    }

    int create_cnt = 0;
    int max_bin = -1;

    for (int j = 0; j < n_features; j++) {
        double step = 1.0;
        if (self->types[j] == CONTINOUS) {
            step = (to_values[j] - from_values[j]) / self->step;
        }

        max_bin = max((int)((to_values[j] - from_values[j]) / step), max_bin);
    }

    Stump** built_classifiers = malloc(sizeof(Stump*) * ((max_bin + 1) * n_features));
    memset(built_classifiers, 0, sizeof(Stump*) * ((max_bin + 1) * n_features));

    for (int j = 0; j < n_features; j++) {
        double step = 1.0;
        if (self->types[j] == CONTINOUS) {
            step = (to_values[j] - from_values[j]) / self->step;
        }

        int bin_index = 0;
        for (double value = from_values[j]; value < to_values[j]; value += step) {
            Stump* classifier = malloc(sizeof(Stump));
            init_stump(classifier, j, self->types[j], X, Y, n_samples, value);
            built_classifiers[bin_index + j * (max_bin + 1)] = classifier;

            bin_index += 1;
            create_cnt += 1;
        }
    }

    printf("done building classifiers...\n");

    double* last_model_preds = malloc(sizeof(double) * n_samples);
    memset(last_model_preds, 0, sizeof(double) * n_samples);
    double* last_pred = malloc(sizeof(double) * n_samples);

    for (int i = 0; i < self->n_classifiers; i++) {
        if (self->classifiers[i] != NULL) { // continue to build!
            printf("ignored %d classifiers\n", i + 1);

            // concat trained classifier's opinion
            predict(self->classifiers[i], X, last_pred, n_samples);

            for (int num_samp = 0; num_samp < n_samples; num_samp++) {
                last_model_preds[num_samp] += last_pred[num_samp] * self->weights[i];
            }
            
            continue;
        }

        printf("classifier %d done!\n", i);
        Stump* best_classifier = NULL;

        if (i % 10 == 0) {
            double loss_val = loss(self, X, Y, n_samples);
            printf("Loss: %f\n", loss_val);
        }

        double min_error = INFINITY;
        for (int j = 0; j < n_features; j++) {
            if (j % 400 == 0) {
                printf("features %d done!\n", j);
            }

            double step = 1.0;
            if (self->types[j] == CONTINOUS) {
                step = (to_values[j] - from_values[j]) / self->step;
            }

            int bin_index = 0;
            for (double value = from_values[j]; value < to_values[j]; value += step) {
                double current_error = error(self, X, Y, n_samples, built_classifiers[bin_index + j * (max_bin + 1)],
                    last_model_preds);

                if (current_error < min_error) {
                    best_classifier = built_classifiers[bin_index + j * (max_bin + 1)];
                    min_error = current_error;
                }

                bin_index += 1;
            }
        }

        double eps = min_error / loss(self, X, Y, n_samples);

        if (0.5 * log((1 - eps) / eps) == 0) {
            break;
        }

        self->classifiers[i] = best_classifier;
        self->weights[i] = 0.5 * log((1 - eps) / eps);

        // concat new classifier's opinion
        predict(best_classifier, X, last_pred, n_samples);

        for (int num_samp = 0; num_samp < n_samples; num_samp++) {
            last_model_preds[num_samp] += last_pred[num_samp] * self->weights[i];
        }
    }

    int del_cnt = 0;
    for (int i = 0; i <= max_bin; i++) {
        for (int j = 0; j < n_features; j++) {
            bool used = false;

            for (int k = 0; k < self->n_classifiers; k++) {
                if (built_classifiers[i + j * (max_bin + 1)] == self->classifiers[k]) {
                    used = true;
                    break;
                }
            }

            if (!used) {
                free_stump(built_classifiers[i + j * (max_bin + 1)]);
                del_cnt += 1;
            }
        }
    }

    // clean up from_values and to_values
    free(from_values);
    free(to_values);
    free(built_classifiers);
    free(last_model_preds);
    free(last_pred);
}

void Ada_fit(AdaBoost* model, PyObject* X, PyObject* Y) {
    npy_intp* dims = PyArray_DIMS(X);

    double** C_X;
    PyArray_AsCArray(&X, &C_X, PyArray_DIMS(X), 2, NULL);

    double* C_Y;
    PyArray_AsCArray(&Y, &C_Y, PyArray_DIMS(Y), 1, NULL);

    fit(model, C_X, C_Y, (int)dims[0], (int)dims[1]);

    PyArray_Free(X, C_X);
    PyArray_Free(Y, C_Y);
}

void capsule_cleanup(PyObject* capsule) {
    void* memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

PyArrayObject* Ada_predict2(AdaBoost* model, PyObject* X) {
    double** C_X;
    PyArray_AsCArray(&X, &C_X, PyArray_DIMS(X), 2, NULL);

    npy_intp n_samples = PyArray_DIMS(X)[0];
    double* pred = malloc(n_samples * sizeof(double));

    ada_predict(model, C_X, pred, (int)n_samples);

    PyObject* capsule = PyCapsule_New(pred, NULL, capsule_cleanup);

    PyArrayObject* py_pred = (PyArrayObject*)PyArray_SimpleNewFromData(1, &n_samples, NPY_FLOAT64, pred);

    PyArray_SetBaseObject((PyArrayObject*)py_pred, capsule);

    PyArray_Free(X, C_X);

    return py_pred;
}

double sum(double** integral, int x1, int y1, int x2, int y2) {
    double C = integral[x2][y2];
    double A = 0;
    double B = 0;
    double D = 0;

    if (x1 >= 1 && y1 >= 1) {
        A = integral[x1 - 1][y1 - 1];
    }
    if (x1 >= 1) {
        B = integral[x1 - 1][y2];
    }
    if (y1 >= 1) {
        D = integral[x2][y1 - 1];
    }

    return C + A - B - D;
}

double right_from_left(double** integral, int i, int j, int win_size) {
    int mid = (2 * j - win_size) / 2 - ((win_size + 1) % 2);

    double right_part = sum(integral, i - win_size, mid + 1, i - 1, j - 1);
    double left_part = sum(integral, i - win_size, j - win_size, i - 1, mid);

    return right_part - left_part;
}

double below_from_above(double** integral, int i, int j, int win_size) {
    int mid = (2 * i - win_size) / 2 - ((win_size + 1) % 2);

    double below = sum(integral, mid + 1, j - win_size, i - 1, j - 1);
    double above = sum(integral, i - win_size, j - win_size, mid, j - 1);

    return below - above;
}

double left_diagonal(double** integral, int i, int j, int win_size) {
    int mid_i = (2 * i - win_size) / 2 - ((win_size + 1) % 2);
    int mid_j = (2 * j - win_size) / 2 - ((win_size + 1) % 2);

    double x1 = sum(integral, i - win_size, j - win_size, mid_i, mid_j);
    double x4 = sum(integral, mid_i + 1, mid_j + 1, i - 1, j - 1);
    double whole = sum(integral, i - win_size, j - win_size, i - 1, j - 1);

    return 2 * (x1 + x4) - whole;
}

double line(double** integral, int i, int j, int win_size) {
    int first_40_percent_j = (int)(j - 0.6 * win_size);
    int first_60_percent_j = (int)(j - 0.4 * win_size);

    double middle_area = sum(integral,
        i - win_size, first_40_percent_j, i - 1, first_60_percent_j);

    double whole = sum(integral, i - win_size, j - win_size, i - 1, j - 1);

    return whole - 2 * middle_area;
}

double* haar(
    double** img,
    int height,
    int width,
    double (*fets_computer)(double**, int, int, int),
    int win_size,
    int stride) {

    double** integral = malloc(height * sizeof(double*));

    for (int i = 0; i < height; i++) {
        integral[i] = malloc(width * sizeof(double));
    }

    double sum = 0;
    for (int i = 0; i < height; i++) {
        sum += img[i][0];
        integral[i][0] = sum;
    }

    sum = 0;
    for (int j = 0; j < width; j++) {
        sum += img[0][j];
        integral[0][j] = sum;
    }

    for (int i = 1; i < height; i++) {
        for (int j = 1l; j < width; j++) {
            integral[i][j] = integral[i][j - 1] + integral[i - 1][j] - integral[i - 1][j - 1] + img[i][j];
        }
    }

    double* fets = malloc(((height - win_size) / stride + 1) * ((width - win_size) / stride + 1)* sizeof(double));

    for (int i = win_size; i < height + 1; i+=stride) {
        for (int j = win_size; j < width + 1; j+=stride) {
            fets[(j - win_size) / stride + (i - win_size) / stride * ((width - win_size) / stride + 1)] = 
                fets_computer(integral, i, j, win_size);
        }
    }

    for (int i = 0; i < height; i++) {
        free(integral[i]);
    }
    free(integral);

    return fets;
}

PyObject* py_haar(PyObject* img, int win_size, int feature_extractor, int stride) {
    double** c_img;
    PyArray_AsCArray(&img, &c_img, PyArray_DIMS(img), 2, PyArray_DescrFromType(NPY_FLOAT64));

    double* fets = NULL;
    if (feature_extractor == LEFT_RIGHT) {
        fets = haar(c_img, (int)PyArray_DIMS(img)[0], (int)PyArray_DIMS(img)[1], right_from_left, win_size, stride);
    }
    else if (feature_extractor == ABOVE_BELOW) {
        fets = haar(c_img, (int)PyArray_DIMS(img)[0], (int)PyArray_DIMS(img)[1], below_from_above, win_size, stride);
    }
    else if (feature_extractor == DIAGONAL) {
        fets = haar(c_img, (int)PyArray_DIMS(img)[0], (int)PyArray_DIMS(img)[1], left_diagonal, win_size, stride);
    }
    else if (feature_extractor == LINE) {
        fets = haar(c_img, (int)PyArray_DIMS(img)[0], (int)PyArray_DIMS(img)[1], line, win_size, stride);
    }

    PyArray_Free(img, c_img);

    npy_intp dims[2] = {
        (PyArray_DIMS(img)[0] - win_size) / stride + 1,
        (PyArray_DIMS(img)[1] - win_size) / stride + 1
    };
    PyObject* py_fets = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, fets);
    PyObject* capsule = PyCapsule_New(fets, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)py_fets, capsule);

    return py_fets;
}

void save_adaboost(AdaBoost* model, char* path, int path_length, char* model_name) {
    char* full_path = malloc(path_length + 100);
    sprintf_s(full_path, path_length + 100, "%s\\%s", path, model_name);
    CreateDirectoryA(full_path, NULL);
    FILE* fptr;

    sprintf_s(full_path, path_length + 100, "%s\\%s\\model_params.data", path, model_name);
    fopen_s(&fptr, full_path, "w");

    char data[300];
    sprintf_s(data, 300, "%d\n%d", model->n_classifiers, model->step);
    fprintf(fptr, data);
    fclose(fptr);

    npy_intp N = model->n_classifiers;
    PyObject* py_types = PyArray_SimpleNewFromData(1, &N, NPY_INT32, model->types);
    PyObject* py_weights = PyArray_SimpleNewFromData(1, &N, NPY_FLOAT64, model->weights);

    sprintf_s(full_path, path_length + 100, "%s\\%s\\types.npy", path, model_name);
    Write_npy(py_types, full_path);
    sprintf_s(full_path, path_length + 100, "%s\\%s\\weights.npy", path, model_name);
    Write_npy(py_weights, full_path);

    for (int i = 0; i < model->n_classifiers; i++) {
        sprintf_s(full_path, path_length + 100, "%s\\%s\\c%d.data", path, model_name, i);
        sprintf_s(data, 300, "%d\n%d\n%d\n%d\n%f",
            model->classifiers[i]->feature_index,
            model->classifiers[i]->label[0],
            model->classifiers[i]->label[1],
            model->classifiers[i]->type,
            model->classifiers[i]->value);

        fopen_s(&fptr, full_path, "w");
        fprintf(fptr, data);
        fclose(fptr);
    }

    free(full_path);
}

AdaBoost* load_adaboost(char* path, int path_length) {
    char* full_path = malloc(path_length + 100);
    char data[300];
    FILE* fptr;

    sprintf_s(full_path, path_length + 100, "%s\\model_params.data", path);
    fopen_s(&fptr, full_path, "r");

    fgets(data, 300, fptr);
    int n_classifiers = atoi(data);
    fgets(data, 300, fptr);
    int step = atoi(data);
    fclose(fptr);

    sprintf_s(full_path, path_length + 100, "%s\\types.npy", path);
    PyObject* types = Read_npy(full_path);
    sprintf_s(full_path, path_length + 100, "%s\\weights.npy", path);
    PyObject* weights = Read_npy(full_path);
    int* c_types;
    PyArray_AsCArray(&types, &c_types, PyArray_DIMS(types), 1, NULL);

    AdaBoost* model = make_adaboost((int)PyArray_DIMS(types)[0], c_types, step, n_classifiers);

    for (int i = 0; i < n_classifiers; i++) {
        model->weights[i] = *((double*)PyArray_GETPTR1(weights, i));
        sprintf_s(full_path, path_length + 100, "%s\\c%d.data", path, i);
        
        if (fopen_s(&fptr, full_path, "r")) {
            break;
        }

        model->classifiers[i] = malloc(sizeof(Stump));
        model->classifiers[i]->label = malloc(2 * sizeof(int));

        fgets(data, 300, fptr);
        model->classifiers[i]->feature_index = atoi(data);

        fgets(data, 300, fptr);
        model->classifiers[i]->label[0] = atoi(data);

        fgets(data, 300, fptr);
        model->classifiers[i]->label[1] = atoi(data);

        fgets(data, 300, fptr);
        model->classifiers[i]->type = atoi(data);

        fgets(data, 300, fptr);
        model->classifiers[i]->value = atof(data);

        fclose(fptr);
    }
    PyArray_Free(types, c_types);

    free(full_path);

    return model;
}

PyObject* build_integral(int height, int width, PyObject* py_img) {
    double** img;
    PyArray_AsCArray(&py_img, &img, PyArray_DIMS(py_img), 2, PyArray_DescrFromType(NPY_FLOAT64));

    double* integral = malloc(height * width * sizeof(double)); // height x width

    double sum = 0;
    for (int i = 0; i < height; i++) {
        sum += img[i][0];
        integral[i * width] = sum;
    }

    sum = 0;
    for (int j = 0; j < width; j++) {
        sum += img[0][j];
        integral[j] = sum;
    }

    for (int i = 1; i < height; i++) {
        for (int j = 1l; j < width; j++) {
            integral[j + i * width] =
                integral[j - 1 + i * width] + integral[j + (i - 1) * width]
                - integral[j - 1 + (i - 1) * width] + img[i][j];
        }
    }

    npy_intp dims[2] = { height, width };
    PyObject* py_integral = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, integral);
    PyObject* capsule = PyCapsule_New(integral, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject*)py_integral, capsule);

    PyArray_Free(py_img, img);

    return py_integral;
}

double get_haar_fet_at(double** integral, int win_size, int feature_extractor, int stride, int x, int y) {
    npy_intp* img_shape = PyArray_DIMS(integral);

    // solve for i, j and feed it to fets_computer
    int term = ((int)img_shape[1] - win_size) / stride;

    int i = x * stride + win_size;
    int j = y * stride + win_size;

    double (*fets_computer)(double**, int, int, int) = NULL;

    if (feature_extractor == LEFT_RIGHT) {
        fets_computer = right_from_left;
    }
    else if (feature_extractor == ABOVE_BELOW) {
        fets_computer = below_from_above;
    }
    else if (feature_extractor == DIAGONAL) {
        fets_computer = left_diagonal;
    }
    else if (feature_extractor == LINE) {
        fets_computer = line;
    }

    return fets_computer(integral, i, j, win_size);
}

void fill_fet_vec(
    int win_size, 
    int stride, 
    int block_size, 
    int shape_1,
    int* indexes, 
    PyObject* full_fets,
    double** integral,
    int indexes_size) {

    for (int i = 0; i < indexes_size; i++) {
        int extractor_id = indexes[i] / block_size;
        int offset_in_block = indexes[i] % block_size;
        int row = offset_in_block / shape_1;
        int col = offset_in_block % shape_1;

        double fet = get_haar_fet_at(integral, win_size, extractor_id, stride, row, col);

        PyArray_SETITEM(full_fets, PyArray_GETPTR2(full_fets, 0, indexes[i]), PyFloat_FromDouble(fet));
    }
}

double** create_fast_integral(PyObject* integral) {
    double** c_integral;
    PyArray_AsCArray(&integral, &c_integral, PyArray_DIMS(integral), 2, NULL);

    return c_integral;
}

void delete_fast_integral(PyObject* integral, double** c_integral) {
    PyArray_Free(integral, c_integral);
}
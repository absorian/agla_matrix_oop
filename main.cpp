// Iskakov Ismagil CS-02
// i.iskakov@innopolis.university
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using std::vector;
using std::istream;
using std::ostream;
using std::cin;
using std::cout;


#define matrix_error(s) cout << "Error: " << s << '\n'

template<typename T>
class augmented_matrix;

template<typename T>
class permutation_matrix;

template<typename T>
class elimination_matrix;

template<typename T>
class matrix_base {
protected:
    vector<vector<T>> arr;
public:
    matrix_base(uint16_t r, uint16_t c) : arr(r, vector<T>(c, 0)) {}

    matrix_base() = default;

    virtual inline uint16_t rows() const {
        return arr.size();
    }

    virtual inline uint16_t columns() const {
        return arr.empty() ? 0 : arr[0].size();
    }

    virtual inline T *operator[](const uint16_t i) {
        return arr[i].data();
    }

    virtual inline const T *operator[](const uint16_t i) const {
        return arr[i].data();
    }

    virtual inline bool is_square() const {
        return rows() == columns();
    }
};

// basic matrix with available operations, as well as augmented matrix functionality
template<typename T>
class matrix : public matrix_base<T> {
protected:
    using matrix_base<T>::arr;

    // column index of the end of every matrix added, main matrix at 0 index
    vector<uint32_t> col_idx;
public:
    using matrix_base<T>::rows;
    using matrix_base<T>::columns;

    matrix(uint16_t r, uint16_t c) : matrix_base<T>(r, c), col_idx(1, c) {}

    matrix() : matrix_base<T>(), col_idx(1, 0) {}

    explicit matrix(const matrix_base<T> &mtx) : matrix() {
        *this = mtx;
    }

    // consist of copying at the end of arr
    void add_augmented(const matrix_base<T> &mtx) {
        if (rows() != mtx.rows()) return;
        uint32_t old_cols = columns_augmented();
        for (int i = 0; i < mtx.rows(); ++i) {
            arr[i].resize(old_cols + mtx.columns());
            for (int j = old_cols; j < arr[i].size(); ++j) {
                arr[i][j] = mtx[i][j - old_cols];
            }
        }
        col_idx.push_back(columns_augmented());
    }

    // if idx = 0, produces the whole matrix as one (accessing the matrix by usual will get only the main part)
    // if idx != 0, gives augmented matrix at idx starting from 1
    augmented_matrix<T> get_augmented(const uint16_t idx = 0) const {
        uint16_t start = idx ? col_idx[idx - 1] : 0;
        uint16_t end = idx ? col_idx[idx] : columns_augmented();
        return augmented_matrix<T>((matrix_base<double> &) *this, start, end - start);
    }

    inline uint16_t augmented_count() const {
        return col_idx.size() - 1;
    }

    // returns column count of the main matrix
    inline uint16_t columns() const override {
        return col_idx[0];
    }

    // returns the whole column size
    virtual inline uint16_t columns_augmented() const {
        return arr.empty() ? 0 : arr[0].size();
    }

    virtual inline void resize(uint16_t r, uint16_t c) {
        col_idx[0] = c;
        arr.resize(r);
        for (int i = 0; i < r; ++i) {
            arr[i].resize(c, 0);
        }
    }

    void normalize_diagonal() {
        for (int i = 0; i < rows(); ++i) {
            if (i >= columns()) break;
            // dividing each row by its pivot to reduce them to 1
            T div = arr[i][i];
            // 0-division prevention
            if (!div) continue;
            for (int j = 0; j < columns_augmented(); ++j) {
                arr[i][j] /= div;
            }
        }
    }

    matrix transpose() const {
        matrix t(columns(), rows());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < columns(); ++j) {
                t[j][i] = arr[i][j];
            }
        }
        return t;
    }

    matrix transpose_augmented() const {
        matrix t(columns_augmented(), rows());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < columns_augmented(); ++j) {
                t[j][i] = arr[i][j];
            }
        }
        return t;
    }

    matrix operator+(const matrix_base<T> &rv) const {
        if (rows() != rv.rows() || columns() != rv.columns()) {
            matrix_error("the dimensional problem occurred");
            return matrix();
        }
        matrix t(rows(), columns());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < columns(); ++j) {
                t[i][j] = arr[i][j] + rv[i][j];
            }
        }
        return t;
    }

    matrix operator-(const matrix_base<T> &rv) const {
        if (rows() != rv.rows() || columns() != rv.columns()) {
            matrix_error("the dimensional problem occurred");
            return matrix();
        }
        matrix t(rows(), columns());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < columns(); ++j) {
                t[i][j] = arr[i][j] - rv[i][j];
            }
        }
        return t;
    }

    matrix operator*(const matrix<T> &rv) const {
        if (columns() != rv.rows()) {
            matrix_error("the dimensional problem occurred");
            return matrix();
        }
        matrix t(rows(), rv.columns());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < rv.columns(); ++j) {

                for (int common = 0; common < columns(); ++common) {
                    t[i][j] += rv[common][j] * arr[i][common];
                }
            }
        }
        for (int a = 1; a <= rv.augmented_count(); ++a) {
            auto aug = rv.get_augmented(a);
            matrix<T> t_aug(rows(), aug.columns());
            for (int i = 0; i < rows(); ++i) {
                for (int j = 0; j < aug.columns(); ++j) {

                    for (int common = 0; common < columns(); ++common) {
                        t_aug[i][j] += aug[common][j] * arr[i][common];
                    }
                }
            }
            t.add_augmented(t_aug);
        }
        return t;
    }

    inline void swap(uint16_t r1, uint16_t r2) {
        std::swap(arr[r1], arr[r2]);
    }

    matrix &operator=(const matrix &rv) {
        if (this == &rv) return *this;
        arr = rv.arr;
        col_idx = rv.col_idx;
        return *this;
    }

    virtual matrix &operator=(const matrix_base<T> &rv) {
        if (this == &rv) return *this;
        resize(rv.rows(), rv.columns());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < columns(); ++j) {
                arr[i][j] = rv[i][j];
            }
        }
        col_idx.resize(1);
        col_idx[0] = rv.columns();
        return *this;
    }

    // helpers for logging
    enum elim_state {
        ELIM_UNKNOWN, ELIM_ELIMINATION, ELIM_PERMUTATION
    };
    using elim_log_func = void (*)(const matrix<T> &, elim_state);

    void direct_eliminate(elim_log_func log = nullptr) {
        for (int i = 0; i < columns(); ++i) {
            double pivot = arr[i][i];
            int pivot_row = i;
            // choosing pivot
            for (int row = i + 1; row < rows(); ++row) {
                if (std::abs(arr[row][i]) > std::abs(pivot)) {
                    pivot = arr[row][i];
                    pivot_row = row;
                }
            }
            // put the pivot on corresponding diagonal
            if (pivot_row != i) {
                *this = permutation_matrix<T>(rows(), i, pivot_row) * *this;

                // without copying
                // permutation_matrix<T>::permutate_inplace(*this, i, pivot_row);

                if (log != nullptr) log(*this, ELIM_PERMUTATION);
            }
            // eliminate everything beneath pivot
            for (int row = i + 1; row < rows(); ++row) {
                if (!arr[row][i] || !arr[i][i]) continue;
                *this = elimination_matrix<T>(*this, row, i) * *this;

                // without copying
                // elimination_matrix<T>::eliminate_inplace(*this, row, i);

                if (log != nullptr) log(*this, ELIM_ELIMINATION);
            }
        }
    }

    void back_eliminate(elim_log_func log = nullptr) {
        for (int i = columns() - 1; i >= 0; --i) {
            double pivot = arr[i][i];
            // choosing pivot
            for (int row = i - 1; row >= 0; --row) {
                if (std::abs(arr[row][i]) > std::abs(pivot)) {
                    pivot = arr[row][i];
                }
            }
            // eliminate everything beneath pivot
            for (int row = i - 1; row >= 0; --row) {
                if (!arr[row][i] || !arr[i][i]) continue;
                *this = elimination_matrix<T>(*this, row, i) * *this;

                // without copying
                // elimination_matrix<T>::eliminate_inplace(*this, row, i);

                if (log != nullptr) log(*this, ELIM_ELIMINATION);
            }
        }
    }
};

// Reference matrix of some part of original &mtx, must be converted to an usual matrix
// to perform any operations besides of accessing single cells
template<typename T>
class augmented_matrix : public matrix_base<T> {
protected:
    uint32_t offset;
    uint16_t cols;
    matrix_base<T> &mtx;

    augmented_matrix(matrix_base<T> &mtx, uint32_t offset, uint16_t columns) : mtx(mtx), offset(offset),
                                                                               cols(columns) {}

    friend class matrix<T>;

public:

    const matrix_base<T> &get_matrix() const {
        return mtx;
    }

    inline uint16_t rows() const override {
        return mtx.rows();
    }

    inline uint16_t columns() const override {
        return cols;
    }

    inline T *operator[](const uint16_t i) override {
        return mtx[i] + offset;
    }

    inline const T *operator[](const uint16_t i) const override {
        return mtx[i] + offset;
    }
};

// name corresponding with the task
template<typename T>
class ColumnVector : public matrix<T> {
protected:
    using matrix<T>::resize;
    using matrix_base<T>::arr;
public:
    using matrix<T>::rows;

    explicit ColumnVector(uint16_t d) : matrix<T>(d, 1) {}

    ColumnVector() : matrix<T>() {}

    explicit ColumnVector(const matrix_base<T> &mtx) {
        *this = mtx;
    }

    virtual inline void resize(uint16_t d) {
        resize(d, 1);
    }

    ColumnVector &operator=(const matrix_base<T> &rv) override {
        if (this == &rv) return *this;
        resize(rv.rows(), 1);
        for (int i = 0; i < rows(); ++i) {
            arr[i][0] = rv[i][0];
        }
        matrix<T>::col_idx.resize(1);
        matrix<T>::col_idx[0] = rv.columns();
        return *this;
    }

    // helper to determine solution type for column vector
    enum solution_type {
        UNKNOWN, UNIQUE, INFINITE, NO_SOLUTION
    };

    solution_type get_solution_type(const matrix<T> &mtx) {
        if (mtx.columns() != rows()) return UNKNOWN;
        bool inf = false;
        for (int i = 0; i < rows(); ++i) {
            bool loose = true;
            // check if it is free row with all zeros
            for (int j = 0; j < mtx.columns(); ++j) {
                if (mtx[i][j]) {
                    loose = false;
                    break;
                }
            }
            // if yes
            if (loose) {
                // but the vector row is non-zero
                if (arr[i][0]) return NO_SOLUTION;
                // otherwise it has infinity solutions
                inf = true;
            }
        }
        if (inf) return INFINITE;
        return UNIQUE;
    }
//    c++ does not allow overriding by return value, which is frustrating
//    inline T &operator[](const uint16_t i) override {
//        return arr[i][0];
//    }
//
//    inline const T& operator[](const uint16_t i) const override {
//        return arr[i][0];
//    }
};

template<typename T>
class square_matrix : public matrix<T> {
protected:
    using matrix<T>::resize;
    using matrix<T>::operator=;

    using matrix_base<T>::arr;
    using matrix<T>::col_idx;
public:
    using typename matrix<T>::elim_state;
    using typename matrix<T>::elim_log_func;

    explicit square_matrix(uint16_t d) : matrix<T>(d, d) {}

    square_matrix() : matrix<T>() {}

    inline uint16_t dimension() const {
        return matrix<T>::rows();
    }

    virtual inline void resize(uint16_t d) {
        resize(d, d);
    }

    square_matrix &operator=(const square_matrix &rv) {
        if (this == &rv) return *this;
        arr = rv.arr;
        col_idx = rv.col_idx;
        return *this;
    }

    // determinant that returns std::pair with the value and upper-triangular matrix
    std::pair<double, square_matrix<T>> determinant(elim_log_func log = nullptr) const {
        square_matrix<T> a = *this;
        double det = 1;
        for (int i = 0; i < a.dimension(); ++i) {
            double pivot = a[i][i];
            int pivot_row = i;
            for (int row = i + 1; row < a.dimension(); ++row) {
                if (std::abs(a[row][i]) > std::abs(pivot)) {
                    pivot = a[row][i];
                    pivot_row = row;
                }
            }
            if (pivot == 0) {
                return std::make_pair(0, a);
            }
            if (pivot_row != i) {
                a = permutation_matrix<T>(a.rows(), i, pivot_row) * a;

                // without copying
                // permutation_matrix<T>::permutate_inplace(a, i, pivot_row);
                det *= -1;
                if (log != nullptr) log(a, elim_state::ELIM_PERMUTATION);
            }
            det *= pivot;

            for (int row = i + 1; row < a.dimension(); ++row) {
                a = elimination_matrix<T>(a, row, i) * a;

                // without copying
                // elimination_matrix<T>::eliminate_inplace(a, row, i);
                if (log != nullptr) log(a, elim_state::ELIM_ELIMINATION);
            }
        }
        return std::make_pair(det, a);
    }
};

template<typename T>
class identity_matrix : public square_matrix<T> {
protected:
    using square_matrix<T>::operator=;
    using matrix_base<T>::arr;
public:
    explicit identity_matrix(uint16_t d) : square_matrix<T>(d) {
        for (int i = 0; i < d; ++i) {
            arr[i][i] = 1;
        }
    }

    identity_matrix() : square_matrix<T>() {}

    inline void resize(uint16_t d) override {
        square_matrix<T>::resize(d);
        for (int i = 0; i < d; ++i) {
            arr[i][i] = 1;
        }
    }

    identity_matrix &operator=(const identity_matrix &rv) {
        if (this == &rv) return *this;
        arr = rv.arr;
        return *this;
    }

    inline const T *operator[](const uint16_t i) const override {
        return arr[i].data();
    }
};

template<typename T>
class elimination_matrix : public identity_matrix<T> {
protected:
    using matrix<T>::arr;
public:
    explicit elimination_matrix(uint16_t d) : identity_matrix<T>(d) {}

    elimination_matrix(const matrix<T> &a, uint16_t r, uint16_t c)
            : elimination_matrix(a.rows()) {
        arr[r][c] = -a[r][c] / a[c][c];
    }

    inline void eliminate(const square_matrix<T> &a, uint16_t r, uint16_t c) {
        arr[r][c] = -a[r][c] / a[c][c];
    }

    static void eliminate_inplace(matrix<T> &a, uint16_t r, uint16_t c) {
        T mul = -a[r][c] / a[c][c];
        for (int i = 0; i < a.columns_augmented(); ++i) {
            a[r][i] += mul * a[c][i];
        }
    }
};

template<typename T>
class permutation_matrix : public identity_matrix<T> {
protected:
    using matrix<T>::arr;
public:
    explicit permutation_matrix(uint16_t d) : identity_matrix<T>(d) {}

    permutation_matrix(uint16_t d, uint16_t r1, uint16_t r2) : identity_matrix<T>(d) {
        permutate(r1, r2);
    }

    inline void permutate(uint16_t r1, uint16_t r2) {
        matrix<T>::swap(r1, r2);
    }

    static void permutate_inplace(matrix<T> &a, uint16_t r1, uint16_t r2) {
        a.swap(r1, r2);
    }
};

template<typename T>
ostream &operator<<(ostream &out, const matrix_base<T> &m) {
    double thr = 0.5;
    for (int i = 0; i < out.precision(); ++i) thr *= 0.1;
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.columns(); ++j) {
            if (j > 0) cout << ' ';

            if (std::abs(m[i][j]) < thr) cout << 0.0;
            else cout << m[i][j];

        }
        cout << '\n';
    }
    return out;
}

template<typename T>
istream &operator>>(istream &in, matrix<T> &m) {
    uint16_t r, c;
    cin >> r >> c;
    m.resize(r, c);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.columns(); ++j) {
            cin >> m[i][j];
        }
    }
    return in;
}

template<typename T>
istream &operator>>(istream &in, square_matrix<T> &m) {
    uint16_t d;
    cin >> d;
    m.resize(d);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.columns(); ++j) {
            cin >> m[i][j];
        }
    }
    return in;
}

template<typename T>
istream &operator>>(istream &in, ColumnVector<T> &m) {
    uint16_t d;
    cin >> d;
    m.resize(d);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.columns(); ++j) {
            cin >> m[i][j];
        }
    }
    return in;
}

template<typename T>
istream &operator>>(istream &in, identity_matrix<T> &m) = delete;


// custom log function
template<typename T>
void log(const matrix<T> &a, typename matrix<T>::elim_state state = matrix<T>::ELIM_UNKNOWN) {
    static uint32_t step = 1;
    cout << "step #" << step++ << ":";
    switch (state) {
        case matrix<T>::ELIM_ELIMINATION:
            cout << " elimination\n";
            break;
        case matrix<T>::ELIM_PERMUTATION:
            cout << " permutation\n";
            break;
        default:
            cout << "\n";
            break;
    }
    cout << a.get_augmented();
}

int main() {
    cout << std::fixed << std::setprecision(4);
    int m, n;
    cin >> m;
    int tb[m][2];
    for (int i = 0; i < m; ++i) {
        cin >> tb[i][0] >> tb[i][1];
    }
    cin >> n;
    matrix<double> a(m, n + 1);
    ColumnVector<double> b(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= n; ++j) {
            a[i][j] = std::pow(tb[i][0], j);
        }
        b[i][0] = tb[i][1];
    }
    cout << "A:\n" << a;
    matrix<double> a_t = a.transpose();
    a = a_t * a;
    cout << "A_T*A:\n" << a;
    a.add_augmented(identity_matrix<double>(a.rows()));
    a.direct_eliminate();
    a.back_eliminate();
    a.normalize_diagonal();
    a = a.get_augmented(1);
    cout << "(A_T*A)^-1:\n" << a;
    b = a_t * b;
    cout << "A_T*b:\n" << b;
    cout << "x~:\n" << a * b;
    return 0;
}
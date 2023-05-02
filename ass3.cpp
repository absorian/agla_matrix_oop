// Iskakov Ismagil CS-02
// i.iskakov@innopolis.university
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <random>
#include <complex>

using std::complex;

using std::vector;
using std::istream;
using std::ostream;
using std::cin;
using std::cout;

template<typename>
struct __is_complex_helper : public std::integral_constant<bool, false> {
};

template<typename T>
struct __is_complex_helper<std::complex<T>> : public std::integral_constant<bool, true> {
};

template<typename T> // nice
struct is_complex : __is_complex_helper<T> {
};


#define matrix_error(s) cout << "Error: " << s << '\n'

template<typename T>
class augmented_matrix;

template<typename T>
class identity_matrix;

template<typename T>
class permutation_matrix;

template<typename T>
class elimination_matrix;

template<typename T>
class matrix_base {
protected:
    vector<vector<T>> arr;
public:
    matrix_base() = default;

    matrix_base(uint16_t r, uint16_t c) : arr(r, vector<T>(c, 0)) {}

    matrix_base(const std::initializer_list<vector<T>> &init) : arr(init) {} // TODO: check for rectangleness

    template<class Tp>
    matrix_base(const matrix_base<Tp>& rv) : matrix_base(rv.rows(), rv.columns()) {
        for (int i = 0; i < rv.rows(); ++i) {
            for (int j = 0; j < rv.columns(); ++j) {
                arr[i][j] = rv[i][j];
            }
        }
    }

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

    virtual inline void resize(uint16_t r, uint16_t c) {
        arr.resize(r);
        for (int i = 0; i < r; ++i) {
            arr[i].resize(c, 0);
        }
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

    matrix(const std::initializer_list<vector<T>> &init) : matrix_base<T>(init), col_idx(1, 0) {
        col_idx[0] = init.size() == 0 ? 0 : init.begin()->size();
    }

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
        return augmented_matrix<T>((matrix_base<T> &) *this, start, end - start);
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

    inline void resize(uint16_t r, uint16_t c) override {
        col_idx[0] = c;
        matrix_base<T>::resize(r, c);
    }

    void normalize_diagonal() {
        for (int i = 0; i < rows(); ++i) {
            if (i >= columns()) break;
            // dividing each row by its pivot to reduce them to 1
            T div = arr[i][i];
            // 0-division prevention
            if (div == T{}) continue;
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

    template<class Tp>
    matrix operator+(const matrix_base<Tp> &rv) const {
        static_assert(std::is_arithmetic<Tp>::value || is_complex<Tp>::value, "Type is not arithmetic");
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

    template<class Tp>
    matrix operator-(const matrix_base<Tp> &rv) const {
        static_assert(std::is_arithmetic<Tp>::value || is_complex<Tp>::value, "Type is not arithmetic");
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

    template<class Tp>
    matrix operator*(const matrix<Tp> &rv) const {
        static_assert(std::is_arithmetic<Tp>::value || is_complex<Tp>::value, "Type is not arithmetic");
        if (columns() != rv.rows()) {
            matrix_error("the dimensional problem occurred");
            return matrix();
        }
        matrix t(rows(), rv.columns());
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < rv.columns(); ++j) {

                for (int common = 0; common < columns(); ++common) {
                    t[i][j] += arr[i][common] * T(rv[common][j]);
                }
            }
        }
        for (int a = 1; a <= rv.augmented_count(); ++a) {
            auto aug = rv.get_augmented(a);
            matrix<T> t_aug(rows(), aug.columns());
            for (int i = 0; i < rows(); ++i) {
                for (int j = 0; j < aug.columns(); ++j) {

                    for (int common = 0; common < columns(); ++common) {
                        t_aug[i][j] += arr[i][common] * T(aug[common][j]);
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
            T pivot = arr[i][i];
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
                if (arr[row][i] == T{} || arr[i][i] == T{}) continue;
                *this = elimination_matrix<T>(*this, row, i) * *this;

                // without copying
                // elimination_matrix<T>::eliminate_inplace(*this, row, i);

                if (log != nullptr) log(*this, ELIM_ELIMINATION);
            }
        }
    }

    void back_eliminate(elim_log_func log = nullptr) {
        for (int i = columns() - 1; i >= 0; --i) {
            T pivot = arr[i][i];
            // choosing pivot
            for (int row = i - 1; row >= 0; --row) {
                if (std::abs(arr[row][i]) > std::abs(pivot)) {
                    pivot = arr[row][i];
                }
            }
            // eliminate everything beneath pivot
            for (int row = i - 1; row >= 0; --row) {
                if (arr[row][i] == T{} || arr[i][i] == T{}) continue;
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

template<typename T>
class column_vector : public matrix<T> {
protected:
    using matrix<T>::resize;
    using matrix_base<T>::arr;
public:
    using matrix<T>::rows;

    explicit column_vector(uint16_t d) : matrix<T>(d, 1) {}

    column_vector() : matrix<T>() {}

    column_vector(const std::initializer_list<vector<T>> &init) : matrix<T>(init) {}

    column_vector(const matrix_base<T> &mtx) {
        *this = mtx;
    }

    double norm() {
        double ans = 0;
        for (int i = 0; i < rows(); ++i) {
            ans += arr[i][0] * arr[i][0];
        }
        return sqrt(ans);
    }

    virtual inline void resize(uint16_t d) {
        resize(d, 1);
    }

    column_vector &operator=(const matrix_base<T> &rv) override {
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

    square_matrix() : matrix<T>() {}

    explicit square_matrix(uint16_t d) : matrix<T>(d, d) {}

    square_matrix(const matrix_base<T> &rv) : square_matrix<T>() {
        *this = rv;
    }

    square_matrix(const std::initializer_list<vector<T>> &init) : matrix<T>(init) {}

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

    square_matrix &operator=(const matrix_base<T> &rv) override {
        if (!rv.is_square()) {
            matrix_error("Square matrix must have equal count of rows and columns");
            return *this;
        }
        matrix<T>::operator=(rv);
        return *this;
    }

    square_matrix inverse() {
        square_matrix<T> t = *this;
        t.add_augmented(identity_matrix<T>(t.dimension()));
        t.direct_eliminate();
        t.back_eliminate();
        t.normalize_diagonal();
        return square_matrix<T>(t.get_augmented(1));
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
istream &operator>>(istream &in, column_vector<T> &m) {
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

#define PLOT 1 // 0 - no plot, 1 - v(k), 2 - v(t) & k(t)

#if PLOT > 0
#ifdef WIN32
#define GNUPLOT_NAME "\"C:\\Program Files\\gnuplot\\bin\\gnuplot\" -persist"
#else
#define GNUPLOT_NAME "gnuplot -persist"
#endif
#endif

int main() {
#if PLOT > 0
#ifdef WIN32
    FILE *pipe = _popen(GNUPLOT_NAME, "w");
#else
    FILE* pipe = popen(GNUPLOT_NAME, "w");
#endif
#endif
    cout << std::fixed << std::setprecision(2);

    column_vector<int> initial(2);
    double a1, b1, a2, b2;
    double time_lim;
    int num_pts;

    cin >> initial[0][0] >> initial[1][0];
    cin >> a1 >> b1 >> a2 >> b2;
    cin >> time_lim >> num_pts;

    // equilibrium coordinates
    const column_vector<double> shift{
            {a2 / b2},
            {a1 / b1}
    };

    // constants for simplicity, obtained from jacobian matrix's opposite diagonal
    const double a = -b1 * a2 / b2;
    const double b = b2 * a1 / b1;

    // +-im_root are eigen-values of jacobian
    const complex<double> im_root = {0, sqrt(-a * b)};

    // made out of column eigen-vectors
    square_matrix<complex<double>> A = {
            {1,           1},
            {im_root / a, -im_root / a}
    };

    // coefficients
    column_vector<complex<double>> cfs = A.inverse() * (initial - shift);

    // ranges for the field in gnuplot
#if PLOT > 0
    double rx_min = INT32_MAX, ry_min = INT32_MAX;
    double rx_max = INT32_MIN, ry_max = INT32_MIN;
#endif

    column_vector<complex<double>> points[num_pts + 1];
    double time_shift = time_lim / num_pts;
    double cur_time = 0;

    cout << "t:\n";
    for (int i = 0; i < num_pts + 1; ++i) {
        cout << cur_time << ' ';

        double degree = im_root.imag() * cur_time;
        points[i] = A * matrix<complex<double>>{ {complex{cos(degree), sin(degree)}, 0}, // cf * e^(eigenv * i * t)
                                                {0, complex{cos(-degree), sin(-degree)}} } * cfs + shift;

#if PLOT > 0
        rx_max = std::max(rx_max, points[i][0][0].real());
        ry_max = std::max(ry_max, points[i][1][0].real());
        rx_min = std::min(rx_min, points[i][0][0].real());
        ry_min = std::min(ry_min, points[i][1][0].real());
#endif

        cur_time += time_shift;
    }
    cout << '\n';

#if PLOT == 1
    double rx_shift = rx_max * 0.1;
    double ry_shift = ry_max * 0.1;
    fprintf(pipe, "set xzeroaxis\nset yzeroaxis\n");
    fprintf(pipe, "set xrange [%lf:%lf]\n", rx_min - rx_shift, rx_max + rx_shift);
    fprintf(pipe, "set yrange [%lf:%lf]\n", ry_min - ry_shift, ry_max + ry_shift);
    fprintf(pipe, "plot '-' using 1:2 with " \
            "linespoints title \"predator(y)-prey(x) model\"\n");
#endif

#if PLOT == 2
    fprintf(pipe, "set xzeroaxis\nset yzeroaxis\n");
    fprintf(pipe, "plot '-' using 1:2 with " \
            "lines title \"prey\", " \
                  "'-' using 1:2 with lines title \"predator\"\n");
    cur_time = 0;
#endif


    cout << "v:\n";
    for (int i = 0; i < num_pts + 1; ++i) {
        cout << points[i][0][0].real() << ' ';
#if PLOT == 2
        fprintf(pipe, "%lf\t%lf\n", cur_time, points[i][0][0].real());
        cur_time += time_shift;
#endif
    }
    cout << '\n';

#if PLOT == 2
    fprintf(pipe, "e\n");
    cur_time = 0;
#endif

    cout << "k:\n";
    for (int i = 0; i < num_pts + 1; ++i) {
        cout << points[i][1][0].real() << ' ';
#if PLOT == 1
        fprintf(pipe, "%lf\t%lf\n", points[i][0][0].real(), points[i][1][0].real());
#elif PLOT == 2
        fprintf(pipe, "%lf\t%lf\n", cur_time, points[i][1][0].real());
        cur_time += time_shift;
#endif
    }
    cout << '\n';

#if PLOT > 0
    fprintf(pipe, "e\n");
    fflush(pipe);

    cin.ignore(10000, '\n');
    cout << "Press enter to exit.";
    cin.ignore(10000, '\n');

#ifdef WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
#endif // PLOT > 0
    return 0;
}

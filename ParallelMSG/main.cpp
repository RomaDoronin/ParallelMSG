#include <vector>
#include <omp.h>
#include <iostream>
#include <ctime>

using namespace std;

struct CRSMatrix
{
    int n; // Число строк в матрице 
    int m; // Число столбцов в матрице 
    int nz; // Число ненулевых элементов в разреженной симметричной матрице, лежащих не ниже главной диагонали 
    vector<double> val; // Массив значений матрицы по строкам 
    vector<int> colIndex; // Массив номеров столбцов 
    vector<int> rowPtr; // Массив индексов начала строк 
};

void SLE_Solver_CRS(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count);

// ===================================================================================================================== //
#include <vector>
using namespace std;

double VectorDifSumVal(double * v1, double * v2, int n)
{
    /*vector<double> resArr(omp_get_num_threads());

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        resArr[omp_get_thread_num()] += abs(v1[i] - v2[i]);
    }

    double res = 0;
    for each (double var in resArr)
    {
        res += var;
    }*/

    double res = 0;

    for (int i = 0; i < n; i++)
    {
        res += abs(v1[i] - v2[i]);
    }

    return res;
}

void VectorSum(double * v1, double * v2, int n, double * vRes)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        vRes[i] = v1[i] + v2[i];
    }
}

void VectorMultConst(double val, double * v1, int n, double * vRes)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        vRes[i] = v1[i] * val;
    }
}	

double VectorScalarMult(double * v1, double * v2, int n)
{
    /*vector<double> resArr(omp_get_num_threads());
    std::cout << "VectorScalarMult omp_get_num_threads(): " << omp_get_num_threads() << endl;

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        resArr[omp_get_thread_num()] += v1[i] * v2[i];
    }

    double res = 0;
    for each (double var in resArr)
    {
        res += var;
    }*/

    double res = 0;

    for (int i = 0; i < n; i++)
    {
        res += v1[i] * v2[i];
    }

    return res;
}

void VectorMultMatrix(CRSMatrix & A, double * v1, int n, double * vRes)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        vRes[i] = 0;
        int jStart = A.rowPtr[i];
        int jEnd = A.rowPtr[i + 1];
        for (int j = jStart; j < jEnd; j++)
        {
            vRes[i] += A.val[j] * v1[A.colIndex[j]];
        }
    }
}

void Vec1CopyToVec2(double * v1, double * v2, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        v2[i] = v1[i];
    }
}

class CGradientMethod
{
private:
    void CulcX(double * x, double alf, double * h, double * culcRes)
    {
        VectorMultConst(alf, h, n, vectorMultConst);
        VectorSum(x, vectorMultConst, n, culcRes);
    }

    void CulcH(double * r1, double bet, double * h, double * culcRes)
    {
        VectorMultConst(bet, h, n, vectorMultConst);
        return VectorSum(r1, vectorMultConst, n, culcRes);
    }

    double CulcAlf(double * r, double * h, CRSMatrix & A)
    {
        VectorMultMatrix(A, h, n, vectorMultMatrix);
        return VectorScalarMult(r, r, n) / VectorScalarMult(vectorMultMatrix, h, n);
    }

    void CulcR(double * r, double alf, double * h, /*CRSMatrix & A,*/ double * culcRes)
    {
        //VectorMultMatrix(A, h, n, vectorMultMatrix);
        VectorMultConst(-alf, vectorMultMatrix, n, vectorMultConst);
        VectorSum(r, vectorMultConst, n, culcRes);
    }

    double CulcBet(double * r, double * r1)
    {
        return VectorScalarMult(r1, r1, n) / VectorScalarMult(r, r, n);
    }

    int n;

    double * vectorMultMatrix;
    double * vectorMultConst;

public:
    CGradientMethod()
    {}

    ~CGradientMethod()
    {
        delete vectorMultMatrix;
        delete vectorMultConst;
    }

    void Solve(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count)
    {
        n = A.n;

        double * r = new double[n];
        double * rPr = new double[n];
        double * h = new double[n];
        double * hPr = new double[n];
        double alf;
        double bet;
        double * xPr = new double[n];

        // Начальные данные
        vectorMultMatrix = new double[n];
        VectorMultMatrix(A, x, n, vectorMultMatrix);
        vectorMultConst = new double[n];
        VectorMultConst(-1, vectorMultMatrix, n, vectorMultConst);
        VectorSum(b, vectorMultConst, n, r);
        VectorSum(b, vectorMultConst, n, h);

        VectorMultMatrix(A, h, n, vectorMultMatrix);
        alf = VectorScalarMult(r, r, n) / VectorScalarMult(vectorMultMatrix, h, n);

        Vec1CopyToVec2(x, xPr, n);
        CulcX(xPr, alf, h, x);
        count = 1;

        while (VectorDifSumVal(x, xPr, n) > eps && max_iter >= count)
        {
            Vec1CopyToVec2(r, rPr, n);
            CulcR(rPr, alf, h, /*A,*/ r);  // R_s
            bet = CulcBet(rPr, r);     // BET_s-1
            Vec1CopyToVec2(h, hPr, n);
            CulcH(r, bet, hPr, h);     // H_s
            alf = CulcAlf(r, h, A);    // ALF_s
            Vec1CopyToVec2(x, xPr, n);
            CulcX(xPr, alf, h, x);     // X_s+1

            count++;
        }

        delete r;
        delete rPr;
        delete h;
        delete hPr;
        delete xPr;
    }
};

void SLE_Solver_CRS(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count)
{
    CGradientMethod cgm;

    cgm.Solve(A, b, eps, max_iter, x, count);
}

// ===================================================================================================================== //
void GenVecWithoutNull(std::vector<double> & vec, int n, int var)
{
    for (int i = 0; i < n; i++)
    {
        int randVal = rand() % var + 1;
        vec.push_back(pow(-1, rand() % 2) * randVal);
    }
}

void GenVecWithoutNull(double * vec, int n, int var)
{
    for (int i = 0; i < n; i++)
    {
        int randVal = rand() % var + 1;
        vec[i] = (pow(-1, rand() % 2) * randVal);
    }
}

double CRSMatrixGetValue(CRSMatrix matrix, int i, int j)
{
    if (j < i)
    {
        int tmp = i;
        i = j;
        j = tmp;
    }

    for (int count = matrix.rowPtr[i]; count < matrix.rowPtr[i + 1]; count++)
    {
        if (matrix.colIndex[count] == j)
            return matrix.val[count];
    }

    return 0;
}

void CRSMatrixSetValue(CRSMatrix & matrix, int i, int j, double value)
{
    int index = matrix.rowPtr[i + 1];

    for (int count = matrix.rowPtr[i]; count < matrix.rowPtr[i + 1]; count++)
    {
        if (j < matrix.colIndex[count])
        {
            index = count;
            break;
        }
        else if (j == matrix.colIndex[count])
        {
            matrix.val[count] = value;
            matrix.colIndex[count] = j;
            return;
        }
    }

    matrix.val.insert(matrix.val.begin() + index, value);
    matrix.colIndex.insert(matrix.colIndex.begin() + index, j);

    for (int rowCount = i + 1; rowCount < matrix.n + 1; rowCount++)
        matrix.rowPtr[rowCount]++;
}

double CRSMatrixGetSumLineElem(CRSMatrix & matrix, int i)
{
    double sum = 0;
    int jStart = matrix.rowPtr[i];
    int jEnd = matrix.rowPtr[i + 1];
    for (int j = jStart; j < jEnd; j++)
    {
        sum += abs(matrix.val[j]);
    }

    return sum;
}

void InitCRSMatrix(CRSMatrix & matrix, unsigned int n, unsigned int nz)
{
    //nz = 10000000;
    matrix.n = n;
    matrix.m = n;
    matrix.nz = nz;
    for (int i = 0; i < n + 1; i++)
    {
        matrix.rowPtr.push_back(0);
    }

    std::vector<double> initVec;
    const int varNum = 10;

    std::cout << "NZ: " << nz << endl;

    // 1. Сгенерировать вектор размерности ((nz - n)/2) без 0
    GenVecWithoutNull(initVec, (nz - n) / 2, varNum);

    // 2. Рандомно разместить его в верхнем треугольнике
    for (int i = 0; i < initVec.size(); i++)
    {
        if (initVec.size() >= 100)
            if ((i % (initVec.size() / 100)) == 0)
                std::cout << "\r" << "Init: " << i / (initVec.size() / 100) + 1 << "%";

        int indexI = rand() % (n - 1);
        int indexJ = rand() % (n - indexI - 1) + indexI + 1;

        CRSMatrixSetValue(matrix, indexI, indexJ, initVec[i]);
        // TODO: Нужно хранить только половину матрицы
        CRSMatrixSetValue(matrix, indexJ, indexI, initVec[i]);
    }

    std::cout << endl;

    // 3. Заполнить Правильно главную диагональ
    for (int i = 0; i < n; i++)
    {
        if (n >= 100)
            if ((i % (n / 100)) == 0)
                std::cout << "\r" << "InitM: " << i / (n / 100) + 1 << "%";

        double sum = CRSMatrixGetSumLineElem(matrix, i);

        if (sum == 0)
            sum = rand() % (varNum) + 1;

        int addition = rand() % (varNum);
        CRSMatrixSetValue(matrix, i, i, sum + addition);
    }
}

void PrintVector(double * vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        std::cout << vec[i] << "	";
    }

    std::cout << endl;
}

void PrintCRSMatrix(CRSMatrix &matrix)
{
    std::cout << endl << "CRSMatrix: " << endl;
    for (int i = 0; i < matrix.n; i++)
    {
        for (int j = 0; j < matrix.n; j++)
        {
            std::cout << CRSMatrixGetValue(matrix, i, j) << "	";
        }
        std::cout << endl;
    }

    /*std::cout << "val: ";
    for each (double val in matrix.val)
    {
        std::cout << val << "	";
    }

    std::cout << endl << "col: ";
    for each (double colIndex in matrix.colIndex)
    {
        std::cout << colIndex << "	";
    }

    std::cout << endl << "row ptr: ";
    for each (double rowPtr in matrix.rowPtr)
    {
        std::cout << rowPtr << "	";
    }*/
    std::cout << endl;
}

bool PRKK(double * res1, double * res2, int n, double accuracy)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > accuracy)
            return false;
    }

    return true;
}

double CuclCoef(int n)
{
    double coef;
    if (n + 1 > 50000)
        coef = 0.001; // 0.01%
    else if (n + 1 > 1000)
        coef = 0.005; // 0.5%
    else if (n + 1 > 100)
        coef = 0.01;  // 1%
    else if (n + 1 > 10)
        coef = 0.1;   // 10%
    else
        coef = 0.7;   // 70%

    return coef;
}

int main()
{
    srand(time(0));

    // Входные данные для тестирования
    const unsigned int start         = 1000;
    const unsigned int finish        = 5000; // MAX = 100 000
    const unsigned int step          = 1000;
    const unsigned int maxThreadNum  = 4;
    const double accuracy            = 0.0000001;
    const unsigned int maxIterSize   = 200;
    const bool toPrintData           = false;

    double * xRef;
    double * b;
    double * x;

    // Начало тестирования
    for (unsigned int n = start; n <= finish; n += step)
    {
        std::cout << "=================================== SIZE : " << n << endl;
        xRef = new double[n];
        b = new double[n];
        x = new double[n];

        CRSMatrix A;
        InitCRSMatrix(A, n, n * n * CuclCoef(n));

        if (toPrintData) PrintCRSMatrix(A);

        for (int countThreadNum = 1; countThreadNum < maxThreadNum + 1; countThreadNum++)
        {
            std::cout << "=================================== COUNT TREAD NUM : " << countThreadNum << endl;
            omp_set_num_threads(countThreadNum);

            for (int count = 0; count < 2; count++)
            {
                std::cout << "=================================== COUNT : " << count << endl;
                // Сгенерировать решение
                GenVecWithoutNull(xRef, n, 10);

                if (toPrintData) { std::cout << "xRef:	";  PrintVector(xRef, n); }

                // Подсчетать по решению вектор b
                VectorMultMatrix(A, xRef, n, b);
                if (toPrintData) { std::cout << "b:	";  PrintVector(b, n); }

                // Задать начальное приближение
                for (int i = 0; i < n; i++)
                {
                    x[i] = 1;
                }

                int countTmp = 0;

                double start_time = omp_get_wtime();
                SLE_Solver_CRS(A, b, accuracy, maxIterSize, x, countTmp);
                if (toPrintData) { std::cout << "x: ";  PrintVector(x, n); }

                std::cout << "Solution Time: " << omp_get_wtime() - start_time << endl;
                std::cout << "Num of Steps: " << countTmp << endl;
                std::cout << "PRKK : "; if (PRKK(x, xRef, n, accuracy)) std::cout << "OK"; else std::cout << "FAILED"; std::cout << endl;
            }
        }

        delete b;
        delete x;
        delete xRef;
    }

    std::cout << "\n\nEnter any number...\n";  int a; cin >> a;

    return 0;
}

public class Matrix {

    @Deprecated
    public static double[][] multTermByTerm(double[][] MatA, double[][] MatB) {
        // Verificar se tem o mesmo tamanho

        int NumLines = MatA.length;
        int NumColumns = MatA[0].length;

        double [][] mult = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                mult[i][j] = MatA[i][j] * MatB[i][j];

        return mult;
    }

    public static double[][] multiplyTermByTerm(double[][] MatA, double[][] MatB) {
        // Verificar se tem o mesmo tamanho

        int NumLines = MatA.length;
        int NumColumns = MatA[0].length;

        double [][] mult = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                mult[i][j] = MatA[i][j] * MatB[i][j];

        return mult;
    }

    public static double[][] getSliceColumns(double[][] mat, int init, int end) {

        double[][] new_mat = new double[mat.length][end - init + 1];

        for(int i = 0; i < mat.length; i++) {
            int k = 0;
            for(int j = init; j <= end; j++) {
                new_mat[i][k] = mat[i][j];
                k++;
            }
        }

        return new_mat;
    }

    public static double[][] getSliceRows(double[][] mat, int init, int end) {

        double[][] new_mat = new double[end - init + 1][mat[0].length];

        int k = 0;
        for(int i = init; i <= end; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                new_mat[k][j] = mat[i][j];
            }
            k++;
        }

        return  new_mat;
    }

    public static double[][] vectorAsMatrixColumn(double[] vec) {

        double[][] mat = new double[vec.length][1];

        for(int i = 0; i < vec.length; i++)
            mat[i][0] = vec[i];

        return mat;
    }

    public static double[][] vectorAsMatrixRow(double[] vec) {

        double[][] mat = new double[1][vec.length];

        for(int i = 0; i < vec.length; i++)
            mat[0][i] = vec[i];

        return mat;
    }


    public static double[][] internalExp(double[][] matrix, double value) {

        int NumLines = matrix.length;
        int NumColumns = matrix[0].length;

        double [][] exp = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                exp[i][j] = Math.pow(matrix[i][j], value);

        return exp;
    }

    @Deprecated
    public static double[][] matrixInternalExp(double[][] Mat, double value) {

        int NumLines = Mat.length;
        int NumColumns = Mat[0].length;

        double [][] exp = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                exp[i][j] = Math.pow(Mat[i][j], value);

        return exp;
    }

    public static void randomMatrix(double[][] matrix) {
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                matrix[i][j] =
                        Math.random() * 0.5 *
                                (Math.random() < 0.5 ? 1 : -1);
    }

    /**
     *
     * Implementacao errada
     *
     * **/
    @Deprecated
    public static double[][] matrixMult(double[][] MatA, double[][] MatB) {

        if (MatA.length == 0 || MatA[0].length == 0
                || MatB.length == 0 || MatB[0].length == 0 ) {
            System.out.println("Matriz Vazia");
        }

        if (MatA[0].length != MatB.length) {
            System.out.println("Matrizes não podem ser multiplicadas");
        }

        int NumLines = MatA.length;
        int NumColumns = MatB[0].length;

        double[][] MatR = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++) {
            for(int j = 0; j < NumColumns; j++) {
                for(int n = 0; n < NumLines || n < NumColumns; n++) {
                    MatR[i][j] += MatA[i][n] * MatB[n][j];
                }
            }
        }

        return MatR;
    }

    public static double[][] multiply(double[][] firstMatrix, double[][] secondMatrix) {

        double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyCell(firstMatrix, secondMatrix, row, col);
            }
        }

        return result;
    }

    private static double multiplyCell(double[][] firstMatrix, double[][] secondMatrix, int row, int col) {

        double cell = 0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
        }
        return cell;
    }

    public static double[][] applyFunction(double[][] matrix, Function function) {

        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                result[i][j] = function.run(matrix[i][j]);

        return result;
    }

    @Deprecated
    public static double matrixInternalSum(double[][] Mat) {

        double sum = 0.0;

        int NumLines = Mat.length;
        int NumColumns = Mat[0].length;

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                sum += Mat[i][j];

        return sum;
    }

    public static double internalSum(double[][] Mat) {

        double sum = 0.0;

        int NumLines = Mat.length;
        int NumColumns = Mat[0].length;

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                sum += Mat[i][j];

        return sum;
    }

    public static double[][] transpose(double[][] matrix) {

        double matrix_t[][] = new double[matrix[0].length][matrix.length];

        for(int i = 0; i < matrix_t.length; i++)
            for(int j = 0; j < matrix_t[i].length; j++)
                matrix_t[i][j] = matrix[j][i];

        return matrix_t;
    }

    public static double[][] sum(double[][] matrix_a, double[][] matrix_b) {

        double new_matrix[][] = new double[matrix_a.length][matrix_a[0].length];

        for(int i = 0; i < new_matrix.length; i++)
            for(int j = 0; j < new_matrix[i].length; j++)
                new_matrix[i][j] = matrix_a[i][j] + matrix_a[i][j];

        return new_matrix;
    }

    public static double[][] multiplyConstant(double[][] matrix, double constant) {

        double new_matrix[][] = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < new_matrix.length; i++)
            for(int j = 0; j < new_matrix[i].length; j++)
                new_matrix[i][j] = matrix[i][j] * constant;

        return new_matrix;
    }

    public static void println(double[][] matrix) {
        for(double[] rows : matrix) {
            System.out.print("[ ");
            for (double column : rows) {
                System.out.print(column + " ");
            }
            System.out.println("]");
        }
    }

    @Deprecated
    public static void printMatrix(double[][] matrix) {
        for(int i = 0; i < matrix.length; i++) {
            System.out.print("[ ");
            for(int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println("]");
        }
    }

    @Deprecated
    public static double[][] matrixMinus(double[][] MatA, double[][] MatB) {
        // Verificar se tem o mesmo tamanho

        int NumLines = MatA.length;
        int NumColumns = MatA[0].length;

        double [][] minus = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                minus[i][j] = MatA[i][j] - MatB[i][j];

        return minus;
    }

    public static double[][] minus(double[][] matrix_a, double[][] matrix_b) {
        // Verificar se tem o mesmo tamanho

        int NumLines = matrix_a.length;
        int NumColumns = matrix_a[0].length;

        double [][] minus = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                minus[i][j] = matrix_a[i][j] - matrix_b[i][j];

        return minus;
    }
}

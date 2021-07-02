public class Matrix {

    public static double[][] multiplyTermByTerm(double[][] matrix_a, double[][] matrix_b) {
        // Verificar se tem o mesmo tamanho

        int NumLines = matrix_a.length;
        int NumColumns = matrix_a[0].length;

        double [][] new_matrix = new double[NumLines][NumColumns];

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                new_matrix[i][j] = matrix_a[i][j] * matrix_b[i][j];

        return new_matrix;
    }

    public static double[][] getSliceColumns(double[][] matrix, int init, int end) {

        double[][] new_matrix = new double[matrix.length][end - init + 1];

        for(int i = 0; i < matrix.length; i++) {
            int k = 0;
            for(int j = init; j <= end; j++) {
                new_matrix[i][k] = matrix[i][j];
                k++;
            }
        }

        return new_matrix;
    }

    public static double[][] getSliceRows(double[][] matrix, int init, int end) {

        double[][] new_matrix = new double[end - init + 1][matrix[0].length];

        int k = 0;
        for(int i = init; i <= end; i++) {
            for(int j = 0; j < matrix[i].length; j++) {
                new_matrix[k][j] = matrix[i][j];
            }
            k++;
        }

        return  new_matrix;
    }

    public static double[][] vectorAsMatrixColumn(double[] vector) {

        double[][] matrix = new double[vector.length][1];

        for(int i = 0; i < vector.length; i++)
            matrix[i][0] = vector[i];

        return matrix;
    }

    public static double[][] vectorAsMatrixRow(double[] vector) {

        double[][] matrix = new double[1][vector.length];

        for(int i = 0; i < vector.length; i++)
            matrix[0][i] = vector[i];

        return matrix;
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

    public static void randomMatrix(double[][] matrix) {
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                matrix[i][j] =
                        Math.random() * 0.5 *
                                (Math.random() < 0.5 ? 1 : -1);
    }

    public static double[][] multiply(double[][] matrix_a, double[][] matrix_b) {

        double[][] result = new double[matrix_a.length][matrix_b[0].length];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyCell(matrix_a, matrix_b, row, col);
            }
        }

        return result;
    }

    private static double multiplyCell(double[][] matrix_a, double[][] matrix_b, int row, int col) {

        double cell = 0;
        for (int i = 0; i < matrix_b.length; i++) {
            cell += matrix_a[row][i] * matrix_b[i][col];
        }
        return cell;
    }

    public static double[][] multiplyConstant(double[][] matrix, double constant) {

        double[][] new_matrix = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < new_matrix.length; i++)
            for(int j = 0; j < new_matrix[i].length; j++)
                new_matrix[i][j] = matrix[i][j] * constant;

        return new_matrix;
    }

    public static double[][] applyFunction(double[][] matrix, Function function) {

        double[][] result = new double[matrix.length][matrix[0].length];

        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j < matrix[i].length; j++)
                result[i][j] = function.run(matrix[i][j]);

        return result;
    }

    public static double internalSum(double[][] matrix) {

        double sum = 0.0;

        int NumLines = matrix.length;
        int NumColumns = matrix[0].length;

        for(int i = 0; i < NumLines; i++)
            for(int j = 0; j < NumColumns; j++)
                sum += matrix[i][j];

        return sum;
    }

    public static double[][] transpose(double[][] matrix) {

        double[][] matrix_t = new double[matrix[0].length][matrix.length];

        for(int i = 0; i < matrix_t.length; i++)
            for(int j = 0; j < matrix_t[i].length; j++)
                matrix_t[i][j] = matrix[j][i];

        return matrix_t;
    }

    public static double[][] sum(double[][] matrix_a, double[][] matrix_b) {

        double[][] new_matrix = new double[matrix_a.length][matrix_a[0].length];

        for(int i = 0; i < new_matrix.length; i++)
            for(int j = 0; j < new_matrix[i].length; j++)
                new_matrix[i][j] = matrix_a[i][j] + matrix_b[i][j];

        return new_matrix;
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

    public static void println(double[][] matrix) {
        for(double[] rows : matrix) {
            System.out.print("[ ");
            for (double column : rows) {
                System.out.print(column + " ");
            }
            System.out.println("]");
        }
    }

    public static void println(double[][] matrix, String name) {
        System.out.println(name + " | num_rows: " + matrix.length + ", num_columns: " + matrix[0].length);
        println(matrix);
    }
}

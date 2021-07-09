import java.io.*;
import java.util.*;

class Main {
    public static final String FINAL_WEIGHT_CLEAN_PATH_FILE = "./src/pesos-modelo-limpo-final.txt";
    public static final String FINAL_WEIGHT_RANDOM_PATH_FILE = "./src/pesos-modelo-aleatorio-final.txt";
    public static final String FINAL_WEIGHT_CLEAN_VALIDATION_PATH_FILE = "./src/pesos-modelo-limpo-validation-final.txt";
    public static final String FINAL_WEIGHT_RANDOM_VALIDATION_PATH_FILE = "./src/pesos-modelo-aleatorio-validation-final.txt";
    public static final String BEGIN_WEIGHT_PATH_FILE = "./src/pesos-modelo-inicio";

    public static final String CLEAN_CHARACTERS = "./src/caracteres-limpo.csv";
    public static final String NOISED_CHARACTERS = "./src/caracteres-ruido.csv";
    public static final String NOISED20_CHARACTERS = "./src/caracteres_ruido20.csv";

    public static final String[] mapIndexToLetter = { "A", "B", "C", "D", "E", "J", "K" };

    public static void writeMatrix(double[][] matrix, BufferedWriter writer) throws IOException {
        for (double[] rows : matrix) {
            for (double column : rows) {
                writer.write(column + ",");
            }
            writer.write("\n");
        }
    }

    public static double[][] readMatrix(BufferedReader reader) throws IOException {
        String line;
        int num_columns = 0;
        List<String[]> list = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            if (line.length() == 0)
                break;
            String[] row = line.split(",");
            num_columns = Math.max(num_columns, row.length);
            list.add(row);
        }

        double[][] matrix = new double[list.size()][num_columns];
        for (int i = 0; i < matrix.length; i++) {
            String[] temp = list.get(i);
            for (int j = 0; j < matrix[i].length; j++)
                matrix[i][j] = Double.parseDouble(temp[j]);
        }

        return matrix;
    }

    public static void printCharacter(double[] data) {
        for (int i = 0; i < data.length; i++) {
            if (i % 7 == 0)
                System.out.println();
            if (data[i] < 0) {
                System.out.print(" ");
            } else {
                System.out.print("\u2588");
            }
        }
        System.out.println();
    }

    public static void test(double[][] dataset, String weightsFilePath) {
        Function activation_function = new BipolarSigmoidFunction();
        Function derivative_activation_function = new BipolarSigmoidFunctionDerivative();

        Model model = MLP.architecture(63, 30, 7, activation_function, derivative_activation_function);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(weightsFilePath));
            double[][] hidden_weight = readMatrix(reader);
            double[][] output_weight = readMatrix(reader);
            model.setHiddenWeight(hidden_weight);
            model.setOutputWeight(output_weight);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }

        double[][] x_test = Matrix.getSliceColumns(dataset, 0, 62);
        double[][] y_test = Matrix.getSliceColumns(dataset, 63, 69);

        // Modelo treiado com ABCDEJK
        double precision = 0.0;
        for (int i = 0; i < x_test.length; i++) {
            printCharacter(x_test[i]);
            NeuronState neuron_state = MLP.forwardfeed(model, x_test[i]);
            Matrix.println(neuron_state.getOutput(), "Output");
            System.out.printf("Predicted: %s | Real value: %s\n",
                    mapIndexToLetter[Main.getIndex(neuron_state.getOutput())],
                    mapIndexToLetter[Main.getIndex(Matrix.vectorAsMatrixRow(y_test[i]))]);
            if (Main.getIndex(neuron_state.getOutput()) == Main.getIndex(Matrix.vectorAsMatrixRow(y_test[i]))) {
                precision += 1.0 / x_test.length;
            }
        }
        System.out.printf("Teste Set Precision = %3.2f%%\n", precision * 100);
    }

    public static void train(double[][] dataset, double[][] test, double threshold, boolean useValidation, String weightsFilePath) {
        Function activation_function = new BipolarSigmoidFunction();
        Function derivative_activation_function = new BipolarSigmoidFunctionDerivative();

        Model model = MLP.architecture(63, 30, 7, activation_function, derivative_activation_function);

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(BEGIN_WEIGHT_PATH_FILE));
            writeMatrix(model.getHiddenWeight(), writer);
            writer.write("\n");
            writeMatrix(model.getOutputWeight(), writer);
            writer.flush();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }
        try {
            BufferedReader reader = new BufferedReader(new FileReader(weightsFilePath));
            double[][] hidden_weight = readMatrix(reader);
            double[][] output_weight = readMatrix(reader);
            model.setHiddenWeight(hidden_weight);
            model.setOutputWeight(output_weight);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        MLP.backpropagation(model, dataset, 0.1, threshold, useValidation);
        Matrix.println(getConfusionMatrix(test, model));

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(weightsFilePath));
            writeMatrix(model.getHiddenWeight(), writer);
            writer.write("\n");
            writeMatrix(model.getOutputWeight(), writer);
            writer.flush();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }
    }

    public static int getIndex(double[][] matrix) {
        int index = 0;
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[i].length; j++)
                index = matrix[i][j] > matrix[0][index] ? j : index;

        return index;
    }

    public static double[] getSliceColumns(double[] vector, int init, int end) {

        double[] new_vector = new double[end - init + 1];

        int k = 0;
        for (int j = init; j <= end; j++) {
            new_vector[k] = vector[j];
            k++;
        }

        return new_vector;
    }

    public static double[][] getConfusionMatrix(double[][] dataset, Model model) {

        int output_size = model.getOutputSize();

        double[][] confusion_matrix = new double[output_size][output_size];

        for (double[] row : dataset) {

            double[] x_test = getSliceColumns(row, 0, row.length - output_size - 1);
            double[] y_test = getSliceColumns(row, row.length - output_size, row.length - 1);

            NeuronState neuron_state = MLP.forwardfeed(model, x_test);

            confusion_matrix[getIndex(Matrix.vectorAsMatrixRow(y_test))][getIndex(neuron_state.getOutput())]++;
        }

        return confusion_matrix;
    }

    public static void main(String[] args) {
        System.out.println("Hello MLP!");

        // Rodando com dados de treino sendo apenas os caraceteres limpos e teste com os
        // caracteeres com ruído
        System.out.println("Clean dataset: ");
        List<double[][]> trainAndTest = getTrainTest();
        double[][] train = trainAndTest.get(0);
        double[][] test = trainAndTest.get(1);
        System.out.println("Clean without validation and early stopping: ");
        train(train, test, 0.001, false, FINAL_WEIGHT_CLEAN_PATH_FILE);
        test(test, FINAL_WEIGHT_CLEAN_PATH_FILE);
        System.out.println("Clean with validation and early stopping: ");
        train(train, test, 0.001, true, FINAL_WEIGHT_CLEAN_VALIDATION_PATH_FILE);
        test(test, FINAL_WEIGHT_CLEAN_VALIDATION_PATH_FILE);

        // Rodando com dados de treino sendo dados misturados de todos os 3 datasets e
        // os dados de teste também
        System.out.println("\nMixed dataset");
        List<double[][]> trainAndTestMixed = getTrainTest(true, 0.75);
        double[][] trainMixed = trainAndTestMixed.get(0);
        double[][] testMixed = trainAndTestMixed.get(1);
        System.out.println("Mixed without validation and early stopping: ");
        train(trainMixed, testMixed, 0.001,false, FINAL_WEIGHT_RANDOM_PATH_FILE);
        test(testMixed, FINAL_WEIGHT_RANDOM_PATH_FILE);
        System.out.println("Mixed with validation and early stopping: ");
        train(trainMixed, testMixed, 0.001,true, FINAL_WEIGHT_RANDOM_VALIDATION_PATH_FILE);
        test(testMixed, FINAL_WEIGHT_RANDOM_VALIDATION_PATH_FILE);
    }

    public static List<double[][]> getTrainTest() {
        return getTrainTest(false, 0.0);
    }

    public static List<double[][]> getTrainTest(boolean mixed, double trainRatio) {
        double[][] train = new double[42][];
        double[][] test = new double[21][];

        try {
            BufferedReader reader = new BufferedReader(new FileReader(CLEAN_CHARACTERS));
            double[][] clean = readMatrix(reader);
            reader.close();

            reader = new BufferedReader(new FileReader(NOISED_CHARACTERS));
            double[][] noised = readMatrix(reader);
            reader.close();

            reader = new BufferedReader(new FileReader(NOISED20_CHARACTERS));
            double[][] noised20 = readMatrix(reader);
            reader.close();

            if (mixed) {
                List<double[]> shuffledDataset = new ArrayList<>();
                shuffledDataset.addAll(Arrays.asList(clean));
                shuffledDataset.addAll(Arrays.asList(noised));
                shuffledDataset.addAll(Arrays.asList(noised20));

                Collections.shuffle(shuffledDataset, new Random(99));
                int trainLength = (int) ((int) shuffledDataset.size() * trainRatio);
                int testLength = shuffledDataset.size() - trainLength;

                train = new double[trainLength][];
                test = new double[testLength][];

                for (int i = 0; i < trainLength; i++)
                    train[i] = shuffledDataset.get(i);
                for (int i = 0; i < testLength; i++)
                    test[i] = shuffledDataset.get(trainLength + i);
            } else {
                train = clean;
                test = new double[noised.length + noised20.length][];
                System.arraycopy(noised, 0, test, 0, noised.length);
                System.arraycopy(noised20, 0, test, noised.length, noised20.length);
            }
        } catch (IOException e) {
            System.out.println("Erro ao ler datasets.");
        }

        return Arrays.asList(train, test);
    }
}
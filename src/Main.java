import java.io.*;
import java.util.ArrayList;
import java.util.List;

class Main {

    public static final String TRAIN_PATH_FILE = "C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\train.txt";
    public static final String WEIGHT_PATH_FILE = "C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\pesos-modelo.txt";
    public static final String WITH_NOISE = "C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\com-ruido.txt";
    public static final String WITH_NOISE_2 = "C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\com-ruido-2.txt";

    public static void writeMatrix(double[][] matrix, BufferedWriter writer) throws IOException {
        for(double[] rows : matrix) {
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
            if(line.length() == 0) break;
            String[] row = line.split(",");
            num_columns = Math.max(num_columns, row.length);
            list.add(row);
        }

        double[][] matrix = new double[list.size()][num_columns];
        for(int i = 0; i < matrix.length; i++) {
            String[] temp = list.get(i);
            for(int j = 0; j < matrix[i].length; j++)
                matrix[i][j] = Double.parseDouble(temp[j]);
        }

        return matrix;
    }

    public static void printCharacter(double[] data) {
        for(int i = 0; i < data.length; i++) {
            if(i % 7 == 0) System.out.println();
            if (data[i] < 0) {
                System.out.print(" ");
            } else {
                System.out.print("\u2588");
            }
        }
        System.out.println();
    }

    public static void test(String test_path) {
        Function activation_function = new BipolarSigmoidFunction();
        Function derivative_activation_function = new BipolarSigmoidFunctionDerivative();

        Model model = MLP.architecture(63, 30, 7, activation_function, derivative_activation_function);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(WEIGHT_PATH_FILE));
            double[][] hidden_weight = readMatrix(reader);
            double[][] output_weight = readMatrix(reader);
            model.setHiddenWeight(hidden_weight);
            model.setOutputWeight(output_weight);
        } catch(Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }

        double[][] dataset;

        try {
            dataset = readMatrix(new BufferedReader(new FileReader(test_path)));
        } catch(Exception e) {
            System.out.println(e.getMessage());
            dataset = new double[1][70];
            System.exit(0);
        }

        dataset = Matrix.getSliceColumns(dataset, 0, 62);

        // Modelo treiado com ABCDEJK
        System.out.println("dataset com ruído");
        for(double[] row : dataset) {
            printCharacter(row);
            NeuronState neuron_state = MLP.forwardfeed(model, row);
            Matrix.println(neuron_state.getOutput(), "Output");
        }
    }

    public static void train(double threshold) {
        Function activation_function = new BipolarSigmoidFunction();
        Function derivative_activation_function = new BipolarSigmoidFunctionDerivative();

        Model model = MLP.architecture(63, 30, 7, activation_function, derivative_activation_function);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(WEIGHT_PATH_FILE));
            double[][] hidden_weight = readMatrix(reader);
            double[][] output_weight = readMatrix(reader);
            model.setHiddenWeight(hidden_weight);
            model.setOutputWeight(output_weight);
        } catch(Exception e) {
            System.out.println(e.getMessage());
        }

        double[][] dataset;

        try {
            dataset = readMatrix(new BufferedReader(new FileReader(TRAIN_PATH_FILE)));
        } catch(Exception e) {
            System.out.println(e.getMessage());
            dataset = new double[1][70];
            System.exit(0);
        }

        MLP.backpropagation(model, dataset, 0.1, threshold);

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(WEIGHT_PATH_FILE));
            writeMatrix(model.getHiddenWeight(), writer);
            writer.write("\n");
            writeMatrix(model.getOutputWeight(), writer);
            writer.flush();
        } catch(Exception e) {
            System.out.println(e.getMessage());
            System.exit(0);
        }
    }

    public static void main(String[] args) {
        System.out.println("Hello MLP!");

        train(0.0001);
        test(WITH_NOISE);
    }
}
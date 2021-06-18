import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class Main {

    public static void old_mlp() {
        Function activation = new BipolarFunction(0);
        Function d_activation = new BipolarFunctionDerivate();

        Model model = MLP.architecture(2, 2, 1, activation, d_activation);

        Matrix.printMatrix(model.getHidden());
        System.out.println();
        Matrix.printMatrix(model.getOutput());
        System.out.println();

        double[] x_train = {1, 1};
        MLP.forward(model, x_train);

        double[][] dataset = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
        };

        MLP.backpropagation(model, dataset, 0.1, 0.001);
    }

    public static void old_mlp_2() {
        Function activation = new BipolarSigmoidFunction();
        Function d_activation = new BipolarSigmoidFunctionDerivative();

        Model model = MLP_test.architecture(1, 4, 1, activation, d_activation);

        double[] x_train = { 0, 0 };
        MLP_test.forwardfeed(model, x_train);

        double[][] dataset = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
        };

        MLP_test.backpropagation(model, dataset, 0.1, 0.001);
    }

    public static void getDataset(String pathToCsv) {

        List<String[]> rowList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(pathToCsv))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineItems = line.split(",");
                rowList.add(lineItems);
            }
        } catch(Exception e){
            // Handle any I/O problems
        }

        String out = "";

        double[][] matrix = new double[rowList.size()][70];
        for(int i = 0; i < rowList.size(); i++) {
            String[] row = rowList.get(i);
            for(int j = 0; j < row.length; j++)
//                out += row[j] + ",";
                matrix[i][j] = Double.parseDouble(row[j]);
        }

//        System.out.println(out);
        Matrix.println(matrix, "dataset");

//        return matrix_double;
    }

    public static void main(String[] args) {
        System.out.println("Hello MLP!");

        Function activation = new BipolarSigmoidFunction();
        Function d_activation = new BipolarSigmoidFunctionDerivative();

//        Function activation = new BipolarFunction(0);
//        Function d_activation = new BipolarFunctionDerivate();

        System.out.println(Double.parseDouble("-1"));

        Model_test model = MLP_Test_2.architecture(2, 3, 1, activation, d_activation);

        double[][] and_gate = {
                { 0, 0, 0},
                { 0, 1, 0},
                { 1, 0, 0},
                { 1, 1, 1}
        };

        double[][] xor_gate = {
                { 0, 0, 1},
                { 0, 1, 0},
                { 1, 0, 0},
                { 1, 1, 1}
        };

        double[][] or_gate = {
                { 0, 0, 0},
                { 0, 1, 1},
                { 1, 0, 1},
                { 1, 1, 1}
        };

//        double[][] dataset = getDataset("C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\caracteres-limpo.csv");
        getDataset("C:\\Users\\BSBCo\\IdeaProjects\\Multi Layer Perceptron\\src\\caracteres-limpo.csv");


//        Matrix.println(dataset, "dataset");

//        MLP_Test_2.backpropagation(model, xor_gate, 0.1, 0.001);

        Matrix.println(model.getHiddenWeigth(), "getHiddenWeigth()");
        Matrix.println(model.getOutputWeigth(), "getOutputWeigth()");

        double[] x_test = { 1, 0 };
        NeuronState state = MLP_Test_2.forwardfeed(model, x_test);
        Matrix.println(state.getOutput(),"Resposta!");

    }


}
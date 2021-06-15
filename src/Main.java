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

    public static void main(String[] args) {
        System.out.println("Hello MLP!");


    }
}
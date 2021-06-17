public class MLP_Test_2 {

    // Primeiro Passo
    public static Model_test architecture(
            int input_size,
            int hidden_size,
            int output_size,
            Function activation,
            Function d_activation
    ) {

        // Inicializando pesos
        System.out.println("\n -> Definindo pesos da camada escondida aleatoriamente");
        double[][] hidden_weight = new double[input_size + 1][hidden_size]; // Extra para p bias
        Matrix.randomMatrix(hidden_weight);
        System.out.println("hidden_weight | num_rows: " + hidden_weight.length + ", num_columns: " + hidden_weight[0].length);
        Matrix.println(hidden_weight);

        System.out.println("\n -> Definindo pesos da camada de Saida aleatoriamente");
        double[][] output_weight = new double[hidden_size + 1][output_size]; // Extra para p bias
        Matrix.randomMatrix(output_weight);
        System.out.println("output_weight | num_rows: " + output_weight.length + ", num_columns: " + output_weight[0].length);
        Matrix.println(output_weight);

        return new Model_test(activation, d_activation, hidden_weight, output_weight);
    }

    public static NeuronState forwardfeed(Model_test model, double[] x_train) {

        // Adicionando adicionando bias valendo 1
        double[] vec_temp = new double[x_train.length + 1];
        for(int i = 0; i < x_train.length; i++)
            vec_temp[i] = x_train[i];
        vec_temp[x_train.length] = 1.0;

        // Transformando o vetor em matriz linha para manter o padrao dos neuronios
        double[][] input_network = Matrix.vectorAsMatrixRow(vec_temp);


        // Hidden layer
        System.out.println("\n -> Construindo Neuronios da Camada Escondida");
        double[][] hidden_network_before = Matrix.multiply(input_network, model.getHiddenWeigth());
        System.out.println("hidden_network_before | num_rows: " + hidden_network_before.length + ", num_columns: " + hidden_network_before[0].length);
        Matrix.println(hidden_network_before);

        System.out.println("\n -> Aplicando funcao de ativacao nos neuronios da Camada Escondida");
        double[][] hidden_network = Matrix.applyFunction(hidden_network_before, model.getActivationFunction());
        System.out.println("hidden_network | num_rows: " + hidden_network.length + ", num_columns: " + hidden_network[0].length);
        Matrix.println(hidden_network);


        // Output layer

        // Adicionando 1 para multiplicar com o bias
        double[][] mat_temp = new double[1][hidden_network[0].length + 1];
        for(int i = 0; i < hidden_network[0].length; i++)
            mat_temp[0][i] = hidden_network[0][i];
        mat_temp[0][hidden_network[0].length] = 1.0;

        System.out.println("\n -> Construindo Neuronios Camada de Saida");
        double[][] output_network_before = Matrix.multiply(mat_temp, model.getOutputWeigth());
        System.out.println("output_network_before | num_rows: " + output_network_before.length + ", num_columns: " + output_network_before[0].length);
        Matrix.println(output_network_before);

        System.out.println("\n -> Aplicando funcao de ativacao nos neuronios da Camada de Saida");
        double[][] output_network = Matrix.applyFunction(output_network_before, model.getActivationFunction());
        System.out.println("output_network | num_rows: " + output_network.length + ", num_columns: " + output_network[0].length);
        Matrix.println(output_network);

        return new NeuronState(hidden_network_before, hidden_network, output_network_before, output_network);
    }

    public static void backpropagation(Model_test model, double[][] dataset, double eta, double threshold) {

        int num_dataset_rows = dataset.length;
        int num_dataset_columns = dataset[0].length;

        double squared_error = 2 * threshold;
        int counter = 0;


        // Definindo conjunto de treinamento
        System.out.println("\n -> Definindo conjunto de treinamento");
        double[][] x_train = Matrix.getSliceColumns(dataset, 0, model.getInputSize() - 1);
        System.out.println("x_train | num_rows: " + x_train.length + ", num_columns: " + x_train[0].length);
        Matrix.println(x_train);

        // Definindo conjunto target
        System.out.println("\n -> Definindo conjunto target");
        double[][] y_train = Matrix.getSliceColumns(dataset, model.getInputSize(), num_dataset_columns - 1);
        System.out.println("y_train | num_rows: " + y_train.length + ", num_columns: " + y_train[0].length);
        Matrix.println(y_train);

        // Condicao de parada
        while(squared_error > threshold) {
            squared_error = 0.0;

            for(int i = 0; i < num_dataset_rows; i++) {

                NeuronState neuron_state = forwardfeed(model, x_train[i]);

                // Calculando erro
                double[][] error = Matrix.minus(Matrix.vectorAsMatrixRow(y_train[i]), neuron_state.getOutput());
                System.out.println("\n -> Calculando erro");
                squared_error += Matrix.internalSum(Matrix.internalExp(error, 2));
                System.out.println("squared_error");
                System.out.println(squared_error);

                // Passo 6

                // Calculando Delta do output (termo de informacao de erro) => δ_k = (t_k − y_k)f′(y in_k)
                System.out.println("\n -> Calculando Delta do output (δ_k = (t_k − y_k)f′(y in_k))");
                double[][] delta_output = Matrix.multiplyTermByTerm(
                                Matrix.applyFunction(
                                        neuron_state.getOutput(),
                                        model.getActivationFunctionDerivate()
                                ), error);
                System.out.println("delta_output | num_rows: " + delta_output.length + ", num_columns: " + delta_output[0].length);
                Matrix.println(delta_output);

                // Separando o Bias dos Neuronios
                double[][] output_weight = Matrix.getSliceRows(model.getOutputWeigth(), 0, model.getHiddenSize() - 1);
                double[][] output_bias_weight = Matrix.getSliceRows(model.getOutputWeigth(), model.getHiddenSize(), model.getHiddenSize());

                System.out.println("\n -> Calculando delta nos pesos do Output => ∆wjk = αδk zj e ∆w0k = αδk");
                // Calculando delta nos pesos do output ∆wjk = αδk zj
                double[][] delta_output_weight = new double[output_weight.length][output_weight[0].length];
                for(int k = 0; k < delta_output_weight.length; k++)
                    for(int l = 0; l <delta_output_weight[k].length; l++)
                        delta_output_weight[k][l] = output_weight[k][l] * delta_output[0][l] * eta;
                Matrix.println(delta_output_weight, "delta_weight_output");

                // Calculando delta nos pesos no bias do output ∆w0k = αδk
                double[][] delta_output_bias_weight = new double[1][output_bias_weight[0].length];
                for(int k = 0; k < delta_output_bias_weight[0].length; k++)
                    delta_output_bias_weight[0][k] = output_bias_weight[0][k] * eta;
                Matrix.println(delta_output_bias_weight, "delta_output_bias_weight");
                // Talvez possa ser substituido pela multiplicacao da constante na matriz


                // Passo 7

                // -> Testes para fazer o delta_hidden => explicar no video
                // Matrix.println(delta_output, "delta_output");
                // Matrix.println(Matrix.transpose(delta_output), "Matrix.transpose(delta_output)");
                // Matrix.println(delta_output_weight, "output_weight");
                // Matrix.println(Matrix.transpose(Matrix.multiply(output_weight, Matrix.transpose(delta_output))), "delta_hidden");

                // Calculando Delta do hidden (termo de informacao de erro) => δ in_j = ∑ δ_k w_jk e δj = δ in_jf′(z_inj)
                System.out.println("\n -> Calculando Delta do Hidden (δ in_j = ∑ δ_k w_jk e δj = δ in_jf′(z_inj))");
                double[][] delta_hidden = Matrix.multiplyTermByTerm(
                        Matrix.applyFunction(
                                neuron_state.getHidden(),
                                model.getActivationFunctionDerivate()
                        ), Matrix.transpose(Matrix.multiply(output_weight, Matrix.transpose(delta_output))));
                Matrix.println(delta_hidden, "delta_hidden");;


                Matrix.println(model.getHiddenWeigth(), "model.getHiddenWeigth()");
                // Separando o Bias dos Neuronios
                double[][] hidden_weight = Matrix.getSliceRows(model.getHiddenWeigth(), 0, model.getInputSize() - 1);
                double[][] hidden_bias_weight = Matrix.getSliceRows(model.getHiddenWeigth(), model.getInputSize(), model.getInputSize());

                System.out.println("\n -> Calculando delta nos pesos do Hidden => ∆vij = αδj xi e ∆v0j = αδj");
                // Calculando delta nos pesos do hidden ∆vij = αδj xi
                double[][] delta_hidden_weight = new double[hidden_weight.length][hidden_weight[0].length];
                for(int k = 0; k < delta_hidden_weight.length; k++)
                    for(int l = 0; l <delta_hidden_weight[k].length; l++)
                        delta_hidden_weight[k][l] = hidden_weight[k][l] * delta_hidden[0][l] * eta;
                Matrix.println(delta_hidden_weight, "delta_hidden_weight");

                // Calculando delta nos pesos no bias do hidden ∆v0j = αδj
                double[][] delta_hidden_bias_weight = new double[1][hidden_bias_weight[0].length];
                for(int k = 0; k < delta_hidden_bias_weight[0].length; k++)
                    delta_hidden_bias_weight[0][k] = hidden_bias_weight[0][k] * eta;
                Matrix.println(delta_hidden_bias_weight, "delta_hidden_bias_weight");
                // Talvez possa ser substituido pela multiplicacao da constante na matriz


                // Passo 8

                // Adicionando Bias Novamente a camada de output
                double[][] mat_temp_output = new double[delta_output_weight.length + 1][delta_output_weight[0].length];
                for(int k = 0; k < delta_output_weight.length; k++)
                    mat_temp_output[k] = delta_output_weight[k];
                mat_temp_output[delta_output_weight.length] = delta_output_bias_weight[0];

                // Atribunindo novos valores para a camada output
                System.out.println("\n -> Atribuindo novos pesos para o output (wjk (new) = wjk (old) + ∆wjk)");
                model.setOutputWeigth(Matrix.sum(model.getOutputWeigth(), mat_temp_output));
                Matrix.println(model.getOutputWeigth(), "model.getOutputWeigth()");


                // Adicionando Bias Novamente a camada hidden
                double[][] mat_temp_hidden = new double[delta_hidden_weight.length + 1][delta_hidden_weight[0].length];
                for(int k = 0; k < delta_hidden_weight.length; k++)
                    mat_temp_hidden[k] = delta_hidden_weight[k];
                mat_temp_hidden[delta_hidden_weight.length] = delta_hidden_bias_weight[0];

                // Atribunindo novos valores para a camada output
                System.out.println("\n -> Atribuindo novos pesos para o hidden (vij (new) = vij (old) + ∆vij)");
                model.setHiddenWeigth(Matrix.sum(model.getHiddenWeigth(), mat_temp_hidden));
                Matrix.println(model.getHiddenWeigth(), "model.getHiddenWeigth()");
            }

            System.out.println("\n -> Calculando erro medio quadrado");
            squared_error = squared_error / num_dataset_rows;
            System.out.println("squared_error");
            System.out.println(squared_error);

            System.out.println("---------------------- Counter " + counter + "----------------------");
            counter++;
        }
    }
}

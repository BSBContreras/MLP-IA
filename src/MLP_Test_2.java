public class MLP_Test_2 {

    // Primeiro Passo
    public static Model architecture(
            int input_size,
            int hidden_size,
            int output_size,
            Function activation,
            Function d_activation
    ) {

        // Inicializando pesos
        System.out.println("\n -> Definindo pesos da camada escondida aleatoriamente");
        double[][] hidden = new double[hidden_size][input_size + 1]; // Extra para p bias
        Matrix.randomMatrix(hidden);
        System.out.println("hidden");
        Matrix.println(hidden);

        System.out.println("\n -> Definindo pesos da camada de Saida aleatoriamente");
        double[][] output = new double[output_size][hidden_size + 1]; // Extra para p bias
        Matrix.randomMatrix(output);
        System.out.println("output");
        Matrix.println(output);

        return new Model(activation, d_activation, hidden, output);
    }


    public static NetworkState forwardfeed(Model model, double[] input) {

        // Adicionando 1 para multiplicar com o bias
        double[] vec_temp = new double[input.length + 1];
        for(int i = 0; i < input.length; i++)
            vec_temp[i] = input[i];
        vec_temp[input.length] = 1.0;

        // Transformando o vetor em matriz coluna
        double[][] x_train = Matrix.vectorAsMatrixColumn(vec_temp);


        // Hidden layer
        System.out.println("\n -> Construindo Camada Escondida");
        double[][] hidden_network = Matrix.multiply(model.getHidden(), x_train);
        System.out.println("hidden_network");
        Matrix.println(hidden_network);

        System.out.println("\n -> Aplicando funcao de ativacao na camana escondida");
        double[][] hidden_network_function = Matrix.applyFunction(hidden_network, model.getActivation());
        System.out.println("hidden_network_function");
        Matrix.println(hidden_network_function);


        // Output layer

        // Adicionando 1 para multiplicar com o bias
        double[][] mat_temp = new double[hidden_network_function.length + 1][1];
        for(int i = 0; i < hidden_network_function.length; i++)
            mat_temp[i][0] = hidden_network_function[i][0];
        mat_temp[hidden_network_function.length][0] = 1.0;

        System.out.println("\n -> Construindo Camada de Saida");
        double[][] output_network = Matrix.multiply(model.getOutput(), mat_temp);
        System.out.println("output_network");
        Matrix.println(output_network);

        System.out.println("\n -> Aplicando funcao de ativacao na camana de Saida");
        double[][] output_network_function = Matrix.applyFunction(output_network, model.getActivation());
        System.out.println("output_network_function");
        Matrix.println(output_network_function);

        return new NetworkState(hidden_network, hidden_network_function, output_network, output_network_function);
    }


    public static void backpropagation(Model model, double[][] dataset, double eta, double threshold) {

        int num_dataset_rows = dataset.length;
        int num_dataset_columns = dataset[0].length;

        double squared_error = 2 * threshold;

        // Definindo conjunto de treinamento
        System.out.println("\n -> Definindo conjunto de treinamento");
        double[][] x_train = Matrix.getSliceColumns(dataset, 0, model.getInputSize() - 1);
        System.out.println("x_train");
        Matrix.println(x_train);

        // Definindo conjunto target
        System.out.println("\n -> Definindo conjunto target");
        double[][] y_train = Matrix.getSliceColumns(dataset, model.getInputSize(), num_dataset_columns - 1);
        System.out.println("y_train");
        Matrix.println(y_train);

        while(squared_error > threshold) {

            squared_error = 0.0;

            for(int i = 0; i < num_dataset_rows; i++) {

                NetworkState network_state = forwardfeed(model, x_train[i]);

                // Calculando erro
                double[][] error = Matrix.minus(network_state.getOutputNetworkFunction(), Matrix.vectorAsMatrixColumn(y_train[i]));
                System.out.println("\n -> Calculando erro");
                squared_error += Matrix.internalSum(Matrix.internalExp(error, 2));
                System.out.println("squared_error");
                System.out.println(squared_error);

                // Calculando Delta do output (termo de informacao de erro ) => δ_k = (t_k − y_k)f′(y in_k)
                System.out.println("\n -> Calculando Delta do output");
                double[][] delta_output =
                        Matrix.multiplyTermByTerm(
                                Matrix.applyFunction(
                                        network_state.getOutputNetworkFunction(),
                                        model.getActivationFunctionDerivate()
                                ), error);
                System.out.println("delta_output");
                Matrix.println(delta_output);

                // Treinando a Camada Escondida
                double[][] hidden_weight = Matrix.getSliceColumns(model.getOutput(), 0, model.getHiddenSize() - 1);
                // Ainda estou tentando enteder => Remove o theta (bias) ?

                // Calculando Delta da Camada Escondida
                System.out.println("\n -> Calculando Delta da Camada Escondida");
                double[][] delta_hidden = Matrix.multiplyTermByTerm(
                        Matrix.applyFunction(
                                network_state.getHiddenNetworkFunction(),
                                model.getActivationFunctionDerivate()
                        ), Matrix.transpose(Matrix.multiply(delta_output, hidden_weight))); // Verificar se posso fazer a Transposta aqui
                System.out.println("delta_hidden");
                Matrix.println(delta_hidden);


                // Treinamento

                // Adiciona o theta na camada escondida
                double[][] mat_temp = new double[network_state.getHiddenNetworkFunction().length + 1][1];
                for(int k = 0; k < network_state.getHiddenNetworkFunction().length; k++)
                    mat_temp[k][0] = network_state.getHiddenNetworkFunction()[k][0];
                mat_temp[network_state.getHiddenNetworkFunction().length][0] = 1.0;

                // Atribuindo novos pesos na camada de Saida
                System.out.println("\n -> Calculando novos peseos para o output");
                double[][] new_output =
                        Matrix.sum(Matrix.transpose(Matrix.multiplyConstant(
                                Matrix.multiply(
                                        mat_temp,
                                        delta_output
                                ),  // Verificar se o mat_temp eo delta_output podem fazer a multiplicacao trocados
                                eta)), // Verificar se posso usar a tranposta aqui
                                model.getOutput()
                        );

                model.setOutput(Matrix.transpose(new_output)); // Testando

                System.out.println("new_output");
                Matrix.println(model.getOutput());


                // Adiciona o theta na camada Escondida
                double[] vec_temp = new double[x_train[i].length + 1];
                for(int k = 0; k < x_train[i].length; k++)
                    vec_temp[k] = x_train[i][k];
                vec_temp[x_train[i].length] = 1.0;
                mat_temp = Matrix.vectorAsMatrixRow(vec_temp);

                // Atribuindo novos pesos para a camada Escondida
                System.out.println("\n -> Calculando novos peseos para o hidden");
                double[][] new_hidden = Matrix.sum(
                        Matrix.transpose(Matrix.multiplyConstant(
                                Matrix.multiply(
                                        delta_hidden,
                                        mat_temp
                                ),  // Verificar se o mat_temp eo delta_output podem fazer a multiplicacao trocados
                                eta)), // Verificar se posso usar a tranposta aqui
                        model.getHidden()
                );

                model.setHidden(Matrix.transpose(new_hidden)); // Testando

                System.out.println("new_hidden");
                Matrix.println(model.getHidden());

            }

            System.out.println("\n -> Calculando erro medio quadrado");
            squared_error = squared_error / num_dataset_rows;
            System.out.println("squared_error");
            System.out.println(squared_error);
        }
    }
}

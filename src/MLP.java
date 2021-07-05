import java.util.*;

public class MLP {

    // Primeiro Passo
    public static Model architecture(
            int input_size,
            int hidden_size,
            int output_size,
            Function activation,
            Function d_activation
    ) {

        // Inicializando pesos
//        System.out.println("\n -> Definindo pesos da camada escondida aleatoriamente");
        double[][] hidden_weight = new double[input_size + 1][hidden_size]; // Extra para p bias
        Matrix.randomMatrix(hidden_weight);
//        Matrix.println(hidden_weight, "hidden_weight");

//        System.out.println("\n -> Definindo pesos da camada de Saida aleatoriamente");
        double[][] output_weight = new double[hidden_size + 1][output_size]; // Extra para p bias
        Matrix.randomMatrix(output_weight);
//        Matrix.println(output_weight, "output_weight");

        return new Model(activation, d_activation, hidden_weight, output_weight);
    }

    public static NeuronState forwardfeed(Model model, double[] x_train) {

        // Adicionando adicionando bias valendo 1
        double[] vec_temp = new double[x_train.length + 1];
        for(int i = 0; i < x_train.length; i++)
            vec_temp[i] = x_train[i];
        vec_temp[x_train.length] = 1.0;

        // Transformando o vetor em matriz linha para manter o padrao dos neuronios
        double[][] input_network = Matrix.vectorAsMatrixRow(vec_temp);


        // Hidden layer
//        System.out.println("\n -> Construindo Neuronios da Camada Escondida");
        double[][] hidden_network_before = Matrix.multiply(input_network, model.getHiddenWeight());
//        Matrix.println(hidden_network_before, "hidden_network_before");

//        System.out.println("\n -> Aplicando funcao de ativacao nos neuronios da Camada Escondida");
        double[][] hidden_network = Matrix.applyFunction(hidden_network_before, model.getActivationFunction());
//        Matrix.println(hidden_network, "hidden_network");


        // Output layer

        // Adicionando 1 para multiplicar com o bias
        double[][] mat_temp = new double[1][hidden_network[0].length + 1];
        for(int i = 0; i < hidden_network[0].length; i++)
            mat_temp[0][i] = hidden_network[0][i];
        mat_temp[0][hidden_network[0].length] = 1.0;

//        System.out.println("\n -> Construindo Neuronios Camada de Saida");
        double[][] output_network_before = Matrix.multiply(mat_temp, model.getOutputWeight());
//        Matrix.println(output_network_before, "output_network_before");

//        System.out.println("\n -> Aplicando funcao de ativacao nos neuronios da Camada de Saida");
        double[][] output_network = Matrix.applyFunction(output_network_before, model.getActivationFunction());
//        Matrix.println(output_network, "output_network");

        return new NeuronState(hidden_network_before, hidden_network, output_network_before, output_network);
    }

    public static void backpropagation(Model model, double[][] dataset, double eta, double threshold) {
        // Separa o dataset em 2 partes: Treino e validação
        List<double[][]> trainValidation = splitTrainValidation(dataset);
        double[][] train = trainValidation.get(0);
        double[][] validation = trainValidation.get(1);

        int num_dataset_rows = train.length;
        int num_dataset_columns = train[0].length;

        double squared_error = 2 * threshold;
        double validationMinSquaredError = Double.MAX_VALUE;
        double precision = 0;
        int counter = 0;

        // Separa o conjunto de validação em features e label
        double[][] x_validation = Matrix.getSliceColumns(validation, 0, model.getInputSize() - 1);
        double[][] y_validation = Matrix.getSliceColumns(validation, model.getInputSize(), num_dataset_columns - 1);

        // Definindo conjunto de treinamento
//        System.out.println("\n -> Definindo conjunto de treinamento");
        double[][] x_train = Matrix.getSliceColumns(train, 0, model.getInputSize() - 1);
//        Matrix.println(x_train, "x_train");

        // Definindo conjunto target
//        System.out.println("\n -> Definindo conjunto target");
        double[][] y_train = Matrix.getSliceColumns(train, model.getInputSize(), num_dataset_columns - 1);
//        Matrix.println(y_train, "y_train");

        // Condicao de parada
        while(squared_error > threshold) {
            squared_error = 0.0;
            precision = 0.0;

            for(int i = 0; i < num_dataset_rows; i++) {

                NeuronState neuron_state = forwardfeed(model, x_train[i]);

                if(Main.getIndex(neuron_state.getOutput()) == Main.getIndex(Matrix.vectorAsMatrixRow(y_train[i]))) {
                    precision += 1.0 / num_dataset_rows;
                }

                // Calculando erro
                double[][] error = Matrix.minus(Matrix.vectorAsMatrixRow(y_train[i]), neuron_state.getOutput());
//                System.out.println("\n -> Calculando erro");
                squared_error += Matrix.internalSum(Matrix.internalExp(error, 2));
//                System.out.println("squared_error");
//                System.out.println(squared_error);

                // Passo 6

                // Calculando Delta do output (termo de informacao de erro) => δ_k = (t_k − y_k)f′(y in_k)
//                System.out.println("\n -> Calculando Delta do output (δ_k = (t_k − y_k)f′(y in_k))");
                double[][] delta_output = Matrix.multiplyTermByTerm(
                                Matrix.applyFunction(
                                        neuron_state.getOutputBefore(),
                                        model.getActivationFunctionDerivative()
                                ), error);
//                Matrix.println(delta_output, "delta_output");

//                System.out.println("\n -> Calculando delta nos pesos do Output => ∆wjk = αδk zj e ∆w0k = αδk");
                // Calculando delta nos pesos do output ∆wjk = αδk zj
                double[][] delta_output_weight = Matrix.multiplyConstant(Matrix.multiply(
                        Matrix.transpose(neuron_state.getHidden()),
                        delta_output
                ), eta);
//                Matrix.println(delta_output_weight, "delta_output_weight");

                // Calculando delta nos pesos no bias do output ∆w0k = αδk
                double[][] delta_output_bias_weight = Matrix.multiplyConstant(
                        delta_output,
                        eta
                );
//                Matrix.println(delta_output_bias_weight, "delta_output_bias_weight");

                // Passo 7

                // Separando o peso do Bias do peso dos Neuronios
                double[][] output_weight = Matrix.getSliceRows(model.getOutputWeight(), 0, model.getHiddenSize() - 1);

                // Calculando Delta do hidden (termo de informacao de erro) => δ in_j = ∑ δ_k w_jk e δj = δ in_jf′(z_inj)
//                System.out.println("\n -> Calculando Delta do Hidden (δ in_j = ∑ δ_k w_jk e δj = δ in_jf′(z_inj))");
                double[][] delta_hidden = Matrix.multiplyTermByTerm(
                        Matrix.applyFunction(
                                neuron_state.getHiddenBefore(),
                                model.getActivationFunctionDerivative()
                        ), Matrix.multiply(delta_output, Matrix.transpose(output_weight)));
//                Matrix.println(delta_hidden, "delta_hidden");;

//                System.out.println("\n -> Calculando delta nos pesos do Hidden => ∆vij = αδj xi e ∆v0j = αδj");
                // Calculando delta nos pesos do hidden ∆vij = αδj xi
                double[][] delta_hidden_weight = Matrix.multiplyConstant(Matrix.multiply(
                        Matrix.vectorAsMatrixColumn(x_train[i]),
                        delta_hidden
                ), eta);
//                Matrix.println(delta_hidden_weight, "delta_hidden_weight");

                double[][] delta_hidden_bias_weight = Matrix.multiplyConstant(
                        delta_hidden,
                        eta
                );
//                Matrix.println(delta_hidden_bias_weight, "delta_hidden_bias_weight");

                // Passo 8

                // Adicionando Bias Novamente a camada de output
                double[][] mat_temp_output = new double[delta_output_weight.length + 1][delta_output_weight[0].length];
                for(int k = 0; k < delta_output_weight.length; k++)
                    mat_temp_output[k] = delta_output_weight[k];
                mat_temp_output[delta_output_weight.length] = delta_output_bias_weight[0];

                // Atribunindo novos valores para a camada output
//                System.out.println("\n -> Atribuindo novos pesos para o output (wjk (new) = wjk (old) + ∆wjk)");
                model.setOutputWeight(Matrix.sum(model.getOutputWeight(), mat_temp_output));
//                Matrix.println(model.getOutputWeigth(), "model.getOutputWeigth()");

                // Adicionando Bias Novamente a camada hidden
                double[][] mat_temp_hidden = new double[delta_hidden_weight.length + 1][delta_hidden_weight[0].length];
                for(int k = 0; k < delta_hidden_weight.length; k++)
                    mat_temp_hidden[k] = delta_hidden_weight[k];
                mat_temp_hidden[delta_hidden_weight.length] = delta_hidden_bias_weight[0];

                // Atribunindo novos valores para a camada output
//                System.out.println("\n -> Atribuindo novos pesos para o hidden (vij (new) = vij (old) + ∆vij)");
                model.setHiddenWeight(Matrix.sum(model.getHiddenWeight(), mat_temp_hidden));
//                Matrix.println(model.getHiddenWeigth(), "model.getHiddenWeigth()");
            }

            // Ao fim de cada época, calcular a acuracia do modelo no conjunto de validação

            double validationSquaredError = 0.0;
            for (int i = 0; i < x_validation.length; i++) {
                NeuronState neuron_state = forwardfeed(model, x_validation[i]);
                double[][] error = Matrix.minus(Matrix.vectorAsMatrixRow(y_validation[i]), neuron_state.getOutput());
                validationSquaredError += Matrix.internalSum(Matrix.internalExp(error, 2));
            }
//            System.out.println("\n -> Calculando erro medio quadrado");
            squared_error = squared_error / num_dataset_rows;
//            System.out.println("squared_error");
            System.out.printf("Epoch = %d", counter);
            System.out.printf(", Squared Error = %.6f", squared_error);
            System.out.printf(", Precision = %3.2f%%", precision * 100);
            System.out.printf(", Validation Squared Error: %.6f\n", validationSquaredError);
            counter++;

            if (validationSquaredError > validationMinSquaredError) {
                System.out.println("Parada antecipada!");
                break;
            }
            validationMinSquaredError = validationSquaredError;
        }
    }

    public static List<double[][]> splitTrainValidation(double[][] dataset) {
        // A ideia é pegar 1 letra para cada conjunto de 7(letras possiveis) dentro do conjunto de teste.
        int validationLength = (int) ((int) dataset.length * 0.3);
        int testLength = dataset.length - validationLength;

        double[][] validation = new double[validationLength][];
        double[][] test = new double[testLength][];

        List<double[]> shuffledDataset = new ArrayList<>(Arrays.asList(dataset));
        Collections.shuffle(shuffledDataset, new Random(99));

        for (int i = 0; i < validationLength; i++) {
            validation[i] = shuffledDataset.get(i);
        }

        for (int i = 0; i < testLength; i++) {
            test[i] = shuffledDataset.get(validationLength+i);
        }

        return Arrays.asList(test, validation);
    }
}

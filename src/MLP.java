class MLP {

    public static Model architecture(
            int input_size,
            int hidden_size,
            int output_size,
            Function activation,
            Function d_activation
    ) {

        double[][] hidden = new double[hidden_size][input_size + 1];
        Matrix.randomMatrix(hidden);

        double[][] output = new double[output_size][hidden_size + 1];
        Matrix.randomMatrix(output);

        return new Model(activation, d_activation, hidden, output);
    }

    public static Forward forward(Model model, double[] x_train) {

        // 11:50 do video 1

        // Transformando vetor em matrix e adicionando o theta no final para multiplicar com outra matriz
        double[][] x_p1 = new double[x_train.length + 1][1];

        int i = 0;
        while(i < x_train.length) {
            x_p1[i][0] = x_train[i];
            i++;
        }
        x_p1[i][0] = 1.0;


        // Hidden layer

        // Multiplica todos os pesos da camada escondida
        double[][] net_h_p = Matrix.matrixMult(model.getHidden(), x_p1);
        System.out.println("\nnet_h_p:");
        Matrix.printMatrix(net_h_p);

        // Aplica a funcao de ativacao pra todo vetor
        double[][] f_net_h_p = new double [net_h_p.length][1];
        for(i = 0; i < net_h_p.length; i++) {
            f_net_h_p[i][0] = model.getActivation().run(net_h_p[i][0]);
        }
        System.out.println("\nf_net_h_p:");
        Matrix.printMatrix(f_net_h_p);


        // Output layer

        double[][] x_p2 = new double[f_net_h_p.length + 1][1];

        i = 0;
        while(i < f_net_h_p.length) {
            x_p2[i][0] = f_net_h_p[i][0];
            i++;
        }
        x_p2[i][0] = 1.0;

        double[][] net_o_p = Matrix.matrixMult(model.getOutput(), x_p2);
        System.out.println("\nnet_o_p:");
        Matrix.printMatrix(net_o_p);

        double[][] f_net_o_p = new double [net_o_p.length][1];
        for(i = 0; i < net_o_p.length; i++) {
            f_net_o_p[i][0] = model.getActivation().run(net_o_p[i][0]);
        }
        System.out.println("\nf_net_o_p:");
        Matrix.printMatrix(f_net_o_p);


        // Retornando Valores
        // [0] -> net_h_p
        // [1] -> f_net_h_p
        // [2] -> net_o_p
        // [3] -> f_net_o_p

        return new Forward(net_h_p, f_net_h_p, net_o_p, f_net_o_p);
    }

    public static void backpropagation(Model model, double[][] dataset, double eta, double threshold) {

        // // Parte 2 do video

        double squared_error = 2 * threshold;
        int counter = 0;

        while(squared_error > threshold) {
            squared_error = 0.0;

            for(int i = 0; i < dataset.length; i++) {

                // Definando set de treinamento
                double[] x_train = new double[model.getInputSize()];
                for(int k = 0; k < x_train.length; k++)
                    x_train[k] = dataset[i][k];


                // Definindo o set Target
                double[] y_train = new double[model.getOutputSize()];
                int l = 0;
                for(int k = model.getInputSize(); k < dataset[0].length; k++){
                    y_train[l] = dataset[i][k];
                    l++;
                }

                Forward results = forward(model, x_train);
                double[][] op = results.getF_net_o_p();

                // Calculando o erro
                double[][] y_p = new double[y_train.length][1];
                for(int k = 0; k < y_train.length; k++)
                    y_p[k][0] = y_train[k];

                double[][] error = Matrix.matrixMinus(y_p, op);
                System.out.println("error");
                Matrix.printMatrix(error);

                squared_error = squared_error + Matrix.matrixInternalSum(Matrix.matrixInternalExp(error, 2));
                System.out.println("squared_error");
                System.out.println(squared_error);

                // Aplica derivada da funcao de ativacao no f_net_o_p
                double[][] temp = new double[results.getF_net_o_p().length][results.getF_net_o_p()[0].length];
                for(int k = 0; k < results.getF_net_o_p().length; k++)
                    for(int n = 0; n < results.getF_net_o_p()[k].length; n++)
                        temp[k][n] = model.getdActivation().run(results.getF_net_o_p()[k][n]);

                System.out.println("Derivada da f_net_o_p");
                Matrix.printMatrix(temp);

                // delta_o_p = error * f'(f_net_o_p)
                double[][] delta_o_p = Matrix.multTermByTerm(error, temp);
                System.out.println("delta_o_p");
                Matrix.printMatrix(delta_o_p);

                // Treinando hidden

                System.out.println("output");
                Matrix.printMatrix(model.getOutput());


                double[][] w_o_kj = Matrix.getSliceColumns(model.getOutput(), 0, model.getHiddenSize());

                System.out.println("w_o_kj");
                Matrix.printMatrix(w_o_kj);

                // Aplica derivada da funcao de ativacao no f_net_h_p
                double[][] temp2 = new double[results.getF_net_h_p().length][results.getF_net_h_p()[0].length];
                for(int k = 0; k < results.getF_net_h_p().length; k++)
                    for(int n = 0; n < results.getF_net_h_p()[k].length; n++)
                        temp2[k][n] = model.getdActivation().run(results.getF_net_h_p()[k][n]);

                System.out.println("temp2: ");
                Matrix.printMatrix(temp2);

                System.out.println("delta_o_p:");
                Matrix.printMatrix(delta_o_p);

                System.out.println("w_o_kj:");
                Matrix.printMatrix(w_o_kj);

                double[][] delta_h_p = Matrix.multTermByTerm(temp2, Matrix.matrixMult(delta_o_p, w_o_kj));

                Matrix.printMatrix((delta_h_p));
                System.exit(55);








            }
        }

    }

}
public class Model_test {

    private Function activation;
    private Function d_activation;
    private double[][] hidden_weigth;
    private double[][] output_weigth;
    private int input_size;
    private int hidden_size;

    public Model_test(Function activation, Function d_activation, double[][] hidden_weigth, double[][] output_weigth) {
        this.activation = activation;
        this.d_activation = d_activation;
        this.hidden_weigth = hidden_weigth;
        this.output_weigth = output_weigth;
        this.input_size = hidden_weigth.length - 1;
        this.hidden_size = output_weigth.length -1;
    }

    public void setHiddenWeigth(double[][] hidden_weigth) {
        this.hidden_weigth = hidden_weigth;
    }

    public void setOutputWeigth(double[][] output_weigth) {
        this.output_weigth = output_weigth;
    }

    public Function getActivationFunction() {
        return activation;
    }

    public Function getActivationFunctionDerivate() {
        return d_activation;
    }

    public double[][] getHiddenWeigth() {
        return hidden_weigth;
    }

    public double[][] getOutputWeigth() {
        return output_weigth;
    }

    public int getInputSize() {
        return input_size;
    }

    public int getHiddenSize() {
        return hidden_size;
    }
}

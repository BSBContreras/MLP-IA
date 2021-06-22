public class Model {

    private final Function activation;
    private final Function d_activation;
    private double[][] hidden_weight;
    private double[][] output_weight;
    private final int input_size;
    private final int hidden_size;

    public Model(Function activation, Function d_activation, double[][] hidden_weight, double[][] output_weight) {
        this.activation = activation;
        this.d_activation = d_activation;
        this.hidden_weight = hidden_weight;
        this.output_weight = output_weight;
        this.input_size = hidden_weight.length - 1;
        this.hidden_size = output_weight.length -1;
    }

    public void setHiddenWeight(double[][] hidden_weight) {
        this.hidden_weight = hidden_weight;
    }

    public void setOutputWeight(double[][] output_weight) {
        this.output_weight = output_weight;
    }

    public Function getActivationFunction() {
        return activation;
    }

    public Function getActivationFunctionDerivative() {
        return d_activation;
    }

    public double[][] getHiddenWeight() {
        return hidden_weight;
    }

    public double[][] getOutputWeight() {
        return output_weight;
    }

    public int getInputSize() {
        return input_size;
    }

    public int getHiddenSize() {
        return hidden_size;
    }
}

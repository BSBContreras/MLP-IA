class Model {

    private Function activation;
    private Function d_activation;
    private double[][] hidden;
    private double[][] output;

    public Model(
            Function activation,
            Function d_activation,
            double[][] hidden,
            double[][] output
    ) {
        this.activation = activation;
        this.d_activation = d_activation;
        this.hidden = hidden;
        this.output = output;
    }

    @Deprecated
    public Function getActivation() {
        return activation;
    }

    public Function getActivationFunction() {
        return activation;
    }

    @Deprecated
    public Function getdActivation() {
        return d_activation;
    }

    public Function getActivationFunctionDerivate() {
        return d_activation;
    }

    public double[][] getHidden() {
        return hidden;
    }

    public double[][] getOutput() {
        return output;
    }

    public void setHidden(double[][] hidden) {
        this.hidden = hidden;
    }

    public void setOutput(double[][] output) {
        this.output = output;
    }

    public int getInputSize() {
        return hidden[0].length - 1;
    }

    public int getOutputSize() {
        return output.length;
    }

    public int getHiddenSize() {
        return hidden.length;
    }
}
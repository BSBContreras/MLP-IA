public class NeuronState {

    private double[][] hidden_before;
    private double[][] hidden;
    private double[][] output_before;
    private double[][] output;

    public NeuronState(double[][] hidden_before, double[][] hidden, double[][] output_before, double[][] output) {
        this.hidden_before = hidden_before;
        this.hidden = hidden;
        this.output_before = output_before;
        this.output = output;
    }

    public double[][] getHiddenBefore() {
        return hidden_before;
    }

    public double[][] getHidden() {
        return hidden;
    }

    public double[][] getOutputBefore() {
        return output_before;
    }

    public double[][] getOutput() {
        return output;
    }
}

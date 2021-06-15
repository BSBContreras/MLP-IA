public class NetworkState {

    private final double[][] hidden_network;
    private final double[][] hidden_network_function;
    private final double[][] output_network;
    private final double[][] output_network_function;

    public NetworkState(double[][] hidden_network, double[][] hidden_network_function, double[][] output_network, double[][] output_network_function) {
        this.hidden_network = hidden_network;
        this.hidden_network_function = hidden_network_function;
        this.output_network = output_network;
        this.output_network_function = output_network_function;
    }

    public double[][] getHiddenNetwork() {
        return hidden_network;
    }

    public double[][] getHiddenNetworkFunction() {
        return hidden_network_function;
    }

    public double[][] getOutputNetwork() {
        return output_network;
    }

    public double[][] getOutputNetworkFunction() {
        return output_network_function;
    }
}

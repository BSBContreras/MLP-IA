class Forward {

    private double[][] net_h_p;
    private double[][] f_net_h_p;
    private double[][] net_o_p;
    private double[][] f_net_o_p;

    public Forward(double[][] net_h_p, double[][] f_net_h_p, double[][] net_o_p, double[][] f_net_o_p) {
        this.net_h_p = net_h_p;
        this.f_net_h_p = f_net_h_p;
        this.net_o_p = net_o_p;
        this.f_net_o_p = f_net_o_p;
    }

    public double[][] getNet_h_p() {
        return net_h_p;
    }

    public double[][] getF_net_h_p() {
        return f_net_h_p;
    }

    public double[][] getNet_o_p() {
        return net_o_p;
    }

    public double[][] getF_net_o_p() {
        return f_net_o_p;
    }
}
class BipolarFunction implements Function {

    private int limiter;

    public BipolarFunction(int limiter) {
        this.limiter = limiter;
    }

    public double run(double x) {
        return x >= limiter ? 1 : -1;
    }
}
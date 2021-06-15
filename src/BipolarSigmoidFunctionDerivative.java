public class BipolarSigmoidFunctionDerivative implements Function {

    @Override
    public double run(double x) {
        return ((2 * Math.exp(x)) / Math.pow(Math.exp(x) + 1, 2));
    }
}

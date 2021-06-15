public class BipolarSigmoidFunction implements Function {

    @Override
    public double run(double x) {

        return (1 - Math.exp(-x)) / (1 + Math.exp(-x));
    }
}

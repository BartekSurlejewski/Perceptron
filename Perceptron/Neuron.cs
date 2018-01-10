using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    class Neuron
    {
        private double[] input;
        private double[] weights;
        private double output { get; set; } = 0;
        private double bias;
        private bool isSigmoid;
        private double sum = 0;
        private double delta { get; set; } = 0; // błąd na neuronie
        private double[] previousWeights;
        private double previousBias;
        private Random random = new Random();

        public Neuron(bool isSigmoid, int inputLength)
        {
            this.isSigmoid = isSigmoid;
            Random random = new Random();
            input = new double[inputLength];
            weights = new double[inputLength];
            previousWeights = new double[inputLength];
            if (this.isSigmoid)
            {
                this.bias = random.NextDouble() - 0.5;
                previousBias = bias;
            }
            else
            {
                this.bias = 0;
            }

            //losowanie wag
            for (int i = 0; i < weights.Count(); i++)
            {
                weights[i] = random.NextDouble() - 0.5;
                previousWeights[i] = weights[i];
            }
        }

        private double Sigmoid()
        {
            return 1.0 / (1.0 + Math.Exp(-this.sum));
        }

        private double SigmoidDerivative()
        {
            return Sigmoid() * (1 - Sigmoid());
        }

        public void DeltaOutput(double expectedValue)
        {
            delta = SigmoidDerivative() * (expectedValue - output);
        }

        public void DeltaHidden(double outputLayerDeltaSum)
        {
            delta = SigmoidDerivative() * outputLayerDeltaSum;
        }

        public void SetInput(double newValue)
        {
            input[0] = newValue;
        }

        public double GetWeight(int index)
        {
            return weights[index];
        }

        public void CalculateSum()
        {
            this.sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                this.sum += input[i] * weights[i];
            }
            if (Program.biasIncluded)
            {
                this.sum = this.sum + this.bias;
            }
        }

        public void CalculateOutputs()
        {
            CalculateSum();
            if (isSigmoid)
            {
                output = Sigmoid();
            }
            else
            {
                output = this.sum;
            }
        }

        public void CalculateWeights()
        {
            double tmpWeight;
            double tmpBias;

            for (int i = 0; i < weights.Length; i++)
            {
                tmpWeight = weights[i];
                weights[i] = weights[i] + Program.stepSize * delta * input[i];
                previousWeights[i] = tmpWeight;
            }

            if (Program.biasIncluded)
            {
                tmpBias = this.bias;
                this.bias = this.bias + Program.stepSize * this.delta;
                previousBias = tmpBias;
            }

        }

        public double GetRandomNumber(double minimum, double maximum)
        {
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        public double GetOutput()
        {
            return output;
        }

        public double GetDelta()
        {
            return delta;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace intAnaliza3
{
    class Neuron
    {
        private double[] input; // x
        private double[] weights; // w
        private double output { get; set; } = 0; //y
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

            //losuj wagi
            for (int i = 0; i < weights.Count(); i++)
            {
                weights[i] = random.NextDouble() - 0.5;
                previousWeights[i] = weights[i];
            }


        }

        //Wartość sigmoidy
        private double Sigmoid()
        {
            return 1.0 / (1.0 + Math.Exp(-this.sum));
        }

        //wartosc pochodnej z sigmoidy
        private double SigmoidDerivative()
        {
            return Sigmoid() * (1 - Sigmoid());
        }

        //błąd wartości warstwy wyjściowej
        public void DeltaOutput(double expectedValue)
        {
            delta = SigmoidDerivative() * (expectedValue - output);
        }

        //błąd wartości warstwy ukrytej
        public void DeltaHidden(double outputLayerDeltaSum)
        {
            delta = SigmoidDerivative() * outputLayerDeltaSum;
        }

        public void SetInput(double newValue, int index)
        {
            input[index] = newValue;
        }

        //pobierz wartosc wag
        public double GetWeight(int index)
        {
            return weights[index];
        }

        //Sumator
        public void CalculateSum()
        {
            this.sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                this.sum = this.sum + input[i] * weights[i];
            }
            //jesli bias jest dolaczony dodaj wartosc z niego
            if (Program.biasIncluded)
            {
                this.sum = this.sum + this.bias;
            }
        }

        //oblicz wartosci wyjsciowe
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

        //oblicz wartosci wag
        public void CalculateWeights()
        {
            double tmpWeight;
            double tmpBias;
            //policz wszystkie 4 wagi ze wzoru
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

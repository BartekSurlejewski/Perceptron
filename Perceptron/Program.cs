using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    class Program
    {
        public static double stepSize = 0.1;    // Współczynnik nauczania
        public static int maxEpochCount = 10000;
        public static bool biasIncluded = true;

        public static int inputNeurons = 5;
        public static int hiddenNeurons = 10;
        public static int outputNeurons = 5;

        static void Main(string[] args)
        {
            double[] input = new double[5];
            double[] output = new double[5];
            double[] result;
            string[] lines = new string[5];

            Random rand = new Random();

            for (int i = 0; i < 5; i++)
            {
                input[i] = rand.NextDouble() * (100.0 - 1.0) + 1.0;
                output[i] = Math.Sqrt(input[i]);
            }

            Network network = new Network(inputNeurons, hiddenNeurons, outputNeurons);

            result = network.train(input, output, maxEpochCount);

            using (System.IO.StreamWriter file = new System.IO.StreamWriter("data.txt", false))
            {
                for (int i = 0; i < 5; i++)
                {
                    file.WriteLine(input[i] + "   " + output[i] + "   " + result[i] + "   " + Math.Abs(output[i] - result[i]));
                }
                file.WriteLine(stepSize);
                file.WriteLine(maxEpochCount);
                file.WriteLine(CalculateGlobalAverageError(output, result));
            }
        }

        private static double CalculateGlobalAverageError(double[] expected, double[] results)
        {
            double counter = 0;

            for (int i = 0; i < expected.Length; i++)
            {
                counter += Math.Abs(expected[i] - results[i]);
            }

            return (counter / expected.Length);
        }
    }
}

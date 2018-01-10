using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace intAnaliza3
{
    class Program
    {
        public static double stepSize = 1;
        public static int maxEpochCount = 100000;
        public static bool biasIncluded = true;

        public static int inputNeurons = 4;
        public static int hiddenNeurons = 2;
        public static int outputNeurons = 4;

        static void Main(string[] args)
        {
            double[][] input = new double[4][];
            double[][] output = new double[4][];
            double[][] test = new double[4][];

            input[0] = output[0] = new double[] { 1,0,0,0};
            input[1] = output[1] = new double[] { 0,1,0,0};
            input[2] = output[2] = new double[] { 0,0,1,0};
            input[3] = output[3] = new double[] { 0,0,0, 1};

            test[0] = new double[] { 1, 0, 0, 0 };
            test[1] = new double[] { 0, 0, 1, 0 };
            test[2] = new double[] { 0, 0, 0, 1 };
            test[3] = new double[] { 0, 1, 0, 0 };

            Network network = new Network(inputNeurons, hiddenNeurons, outputNeurons);

            Console.WriteLine(network.train(input, output, 0.00001, maxEpochCount));
            Console.WriteLine("Test:");
            Console.WriteLine(network.test(test));

            Console.ReadLine();
        } 
    }
}

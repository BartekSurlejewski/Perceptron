using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Perceptron;

using static Perceptron.Neuron;

namespace Perceptron
{
    public class Network
    {
        private Neuron[] inputLayer;
        private Neuron[] hiddenLayer;
        private Neuron[] outputLayer;

        public Network(int inputsNumber, int hiddenNumber, int outputsNumber)
        {
            inputLayer = new Neuron[inputsNumber];
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i] = new Neuron(false, 1);
            }

            hiddenLayer = new Neuron[hiddenNumber];
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i] = new Neuron(true, inputsNumber);
            }

            outputLayer = new Neuron[outputsNumber];
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i] = new Neuron(true, hiddenNumber);
            }
        }

        public double[] train(double[] input, double[] output, int maxEpochCount)
        {
            string finalResults = "";
            bool[] finished = new bool[input.Length];
            int i = 0;
            
            for (i = 0; i < finished.Length; i++)
            {
                finished[i] = false;
            }

            double[] hiddenCopy = new double[hiddenLayer.Length];
            for(i = 0; i<hiddenCopy.Length; i++)
            {
                hiddenCopy[i] = new double();
            }

            double[] outputCopy = new double[input.Length];
            for (i = 0; i < outputLayer.Length; i++)
            {
                outputCopy[i] = new double();
            }

            i = 0;

            while(i < maxEpochCount)
            {
                for (int j = 0; j < input.Length; j++)
                {
                    fPropagate(input);

                    bPropagate(output);
                   
                     hiddenCopy[j] = hiddenLayer[j].GetOutput();


                    outputCopy[j] = outputLayer[j].GetOutput();

                   
                }

                i++;
            }
            for (int index = 0; index < outputCopy.Length; index++)
            {
                outputCopy[index] *= output[index];
            }

            return outputCopy;
        }

        private void fPropagate(double[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                inputLayer[i].SetInput(data[i]);
            }

            calculateInputs(inputLayer);
            transfer(inputLayer, hiddenLayer);

            calculateInputs(hiddenLayer);
            transfer(hiddenLayer, outputLayer);

            calculateInputs(outputLayer);
        }

        private void bPropagate(double[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                outputLayer[i].DeltaOutput(data[i]);
            }
            double deltaSum = 0;
            for(int i2 = 0; i2 < hiddenLayer.Length; i2++)
            {
                deltaSum = 0;
                for(int j = 0; j < outputLayer.Length; j++)
                {
                    deltaSum = deltaSum + outputLayer[j].GetDelta() * outputLayer[j].GetWeight(i2);
                }
                hiddenLayer[i2].DeltaHidden(deltaSum);
            }

            calculateWeights(hiddenLayer);

            calculateWeights(outputLayer);
        }

        private void calculateWeights(Neuron[] layer)
        {
            for(int i = 0; i < layer.Length; i++)
            {
                layer[i].CalculateWeights();
            }
        }

        private void calculateInputs(Neuron[] layer)
        {
            for(int i = 0; i < layer.Length; i++)
            {
                layer[i].CalculateOutputs();
            }
        }

        private void transfer(Neuron[] calculatedLayer, Neuron[] currentLayer)
        {
            for(int i = 0; i < calculatedLayer.Length; i++)
            {
                for(int j = 0; j < currentLayer.Length; j++)
                {
                    currentLayer[j].SetInput(calculatedLayer[i].GetOutput());
                }
            }
        }

        private double CalculateDelta(double[] expected)
        {
            double sum = 0;

            for(int i = 0; i < outputLayer.Length; i++)
            {
                sum += (outputLayer[i].GetOutput() - expected[i]) * (outputLayer[i].GetOutput() - expected[i]);
            }

            return sum / outputLayer.Length;
        }
    }
}

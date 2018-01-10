using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using intAnaliza3;

using static intAnaliza3.Neuron;

namespace intAnaliza3
{
    public class Network
    {
        private Neuron[] inputLayer;
        private Neuron[] hiddenLayer;
        private Neuron[] outputLayer;

        private double networkDelta = 0;

        public Network(int inputsNumber, int hiddenNumber, int outputsNumber)
        {
            //tworzenie warstwy wejsciowej
            inputLayer = new Neuron[inputsNumber];
            for (int i = 0; i < inputLayer.Length; i++)
            {
                // nie sigmoida - false
                inputLayer[i] = new Neuron(false, 1);
            }
            //tworzenie warstwy ukrytej
            hiddenLayer = new Neuron[hiddenNumber];
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                // neuron sigmoida - true
                hiddenLayer[i] = new Neuron(true, inputsNumber);
            }
            //tworzenie warstwy wyjsciowej
            outputLayer = new Neuron[outputsNumber];
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i] = new Neuron(true, hiddenNumber);
            }
        }

        public string train(double[][] input, double[][] output, double finalDelta, int maxEpochCount)
        {
            string finalResults = "";
            bool[] finished = new bool[input.Length];
            int i = 0;
            
            for (i = 0; i < finished.Length; i++)
            {
                finished[i] = false;
            }

            double[][] hiddenCopy = new double[input.Length][];
            for(i = 0; i<hiddenCopy.Length; i++)
            {
                hiddenCopy[i] = new double[hiddenLayer.Length];
            }

            double[][] outputCopy = new double[input.Length][];
            for (i = 0; i < outputLayer.Length; i++)
            {
                outputCopy[i] = new double[outputLayer.Length];
            }

            i = 0;

            while(i < maxEpochCount)
            {
                networkDelta = 0;

                for(int j = 0; j < input.Length; j++)
                {
                    fPropagate(input[j]);

                    bPropagate(output[j]);

                    networkDelta += calculateDelta(output[j]);

                    for(int x = 0; x < hiddenLayer.Length; x++)
                    {
                        hiddenCopy[j][x] = hiddenLayer[x].getOutput();
                    }
                    for (int x = 0; x < outputLayer.Length; x++)
                    {
                        outputCopy[j][x] = outputLayer[x].getOutput();
                    }
                    
                    if(calculateDelta(output[j]) < finalDelta)
                    {
                        finished[j] = true;
                    }
                }
                //Średni błąd
                networkDelta = networkDelta / input.Count();

                if(allFinished(finished))
                {
                    finalResults += "Krok: " + i + " Średnia dokładność: " + networkDelta + "\n";
                    finalResults += allLayersOutput(input.Length, hiddenCopy, outputCopy);

                    networkDelta = 0;
                    return finalResults;
                }

                i++;
            }

            finalResults += "Zakończone na " + Program.maxEpochCount + " kroku z błędem: " + networkDelta + "\n";
            finalResults += allLayersOutput(input.Length, hiddenCopy, outputCopy);
            networkDelta = 0;

            return finalResults;
        }

        public string test(double[][] data)
        {
            string output = "";

            for(int i = 0; i < data.Length; i++)
            {
                output += "Dla wzoru nr " + (i + 1) + "\n";
                fPropagate(data[i]);
                foreach(Neuron neuron in outputLayer)
                {
                    output += neuron.getOutput() + "\n";
                }
            }
            return output;
        }

        public string allLayersOutput(int length, double[][] hiddenCopy, double[][] outputCopy)
        {
            string output = "";
            for(int i = 0; i < length; i++)
            {
                output += "Wzór nr " + (i + 1) + "\n";

                output += "Wyjścia warstwy ukrytej: " + "\n";
                foreach(double value in hiddenCopy[i])
                {
                    output += value + "\n";
                }

                output += "Wyjścia warstwy wyjściowej: " + "\n";
                foreach(double value in outputCopy[i])
                {
                    output += value + "\n";
                }
            }

            return output;
        }

        bool allFinished(bool[] tab)
        {
            foreach(bool value in tab)
            {
                if (value == false)
                    return false;
            }
            return true;
        }

        private void fPropagate(double[] data)
        {
            for(int i = 0; i < data.Length; i++)
            {
                inputLayer[i].setInput(data[i], 0);
            }

            calculateInputs(inputLayer);
            transfer(inputLayer, hiddenLayer);

            calculateInputs(hiddenLayer);
            transfer(hiddenLayer, outputLayer);

            calculateInputs(outputLayer);
        }

        private void bPropagate(double[] data)
        {
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i].DeltaOutput(data[i]);
            }

            double deltaSum = 0;
            for(int i2 = 0; i2 < hiddenLayer.Length; i2++)
            {
                deltaSum = 0;
                for(int j = 0; j < outputLayer.Length; j++)
                {
                    deltaSum = deltaSum + outputLayer[j].getDelta() * outputLayer[j].getWeight(i2);
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
                layer[i].calculateWeights();
            }
        }

        private void calculateInputs(Neuron[] layer)
        {
            for(int i = 0; i < layer.Length; i++)
            {
                layer[i].calculateOutputs();
            }
        }

        private void transfer(Neuron[] calculatedLayer, Neuron[] currentLayer)
        {
            for(int i = 0; i < calculatedLayer.Length; i++)
            {
                for(int j = 0; j < currentLayer.Length; j++)
                {
                    currentLayer[j].setInput(calculatedLayer[i].getOutput(), i);
                }
            }
        }

        private double calculateDelta(double[] expected)
        {
            double sum = 0;

            for(int i = 0; i < outputLayer.Length; i++)
            {
                sum += (outputLayer[i].getOutput() - expected[i]) * (outputLayer[i].getOutput() - expected[i]);
            }

            return sum / outputLayer.Length;
        }
    }
}

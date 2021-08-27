using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetLib
{
    public class FullConnection : ILayer
    {
        public int OutputSize { get; private set; }

        public int InputSize { get; private set; }

        public double[] Output { get; private set; }

        public double[] Input { get; private set; }

        public double[] Errors { get; private set; }

        public double[] Sum { get; private set; }
        public double[] Threshold { get; private set; }
        public double[,] Weights { get; private set; }
        public ActivateFunction Function { get; private set; }
        public FullConnection(double[] input, int outputSize, ActivateFunction func) 
        {
            Input = input;
            InputSize = input.Length;
            OutputSize = outputSize;
            Output = new double[outputSize];
            Errors = new double[outputSize];
            Sum = new double[outputSize];
            Threshold = new double[outputSize];
            Weights = new double[InputSize, OutputSize];
            Function = func;
        }
        public void UpdateResult() 
        {
            for (int o = 0; o < OutputSize; o++) 
            {
                double sum = Threshold[o];
                for (int i = 0; i < InputSize; i++) 
                {
                    sum += Input[i] * Weights[i, o];
                }
                Sum[o] = sum;
                Output[o] = Function.F(sum);
                if (double.IsNaN(Sum[o]))
                    throw new Exception();
            }
        }
        public void UpdateErrors(double[] prevLayerErrors)
        {
            if (prevLayerErrors is null)
                return;
            if (prevLayerErrors.Length != InputSize)
                throw new ArgumentException();
            for (int i = 0; i < InputSize; i++)
            {
                double errorSum = 0;
                for (int o = 0; o < OutputSize; o++) 
                {
                    errorSum += Errors[o] * Weights[i, o];
                }
                prevLayerErrors[i] = errorSum;
            }
        }

        public void Train(double learningRate)
        {
            for (int o = 0; o < OutputSize; o++) 
            {
                double eldf = Function.DF(Sum[o]) * Errors[o] * learningRate;
                if (double.IsNaN(eldf) || double.IsInfinity(eldf))
                    throw new Exception();
                for (int i = 0; i < InputSize; i++) 
                {
                    Weights[i, o] += eldf * Input[i];
                    if (Weights[i, o] > 1000 || Weights[i, o] < -1000)
                        Weights[i, o] = r.NextDouble();
                }
                Threshold[o] += eldf * Threshold[o];
            }
        }
        Random r = new Random();
        public void RandomizeWeights(double range) 
        {
            for (int i = 0; i < InputSize; i++)
                for (int o = 0; o < OutputSize; o++) 
                {
                    Weights[i, o] = (r.NextDouble() - 0.5) * 2;
                }
            for (int o = 0; o < OutputSize; o++)
            {
                Threshold[o] = (r.NextDouble() - 0.5) * range * 2;
            }
        }
    }
}

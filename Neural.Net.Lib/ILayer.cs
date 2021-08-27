using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetLib
{
    public interface ILayer
    {
        public int OutputSize { get; }
        public int InputSize { get; }
        public double[] Output { get; }
        public double[] Input { get; }
        public double[] Errors { get; }
        public void UpdateErrors(double[] prevLayerErrors);
        public void UpdateResult();
        public void Train(double learningRate);
        public void RandomizeWeights(double range);
    }
}

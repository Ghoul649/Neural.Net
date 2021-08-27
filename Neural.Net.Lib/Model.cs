using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace NeuralNetLib
{
    public class Model
    {
        public double[] Input { get; private set; }
        public double[] Output { get; private set; }

        private ILayer[] _layers;
        public IEnumerable<ILayer> Layers { get => _layers; }
        private double[] lastErrors;

        public Model(ILayer[] layers) 
        {
            Input = layers[0].Input;
            Output = layers[layers.Length - 1].Output;
            lastErrors = layers[layers.Length - 1].Errors;
            _layers = layers;
        }
        public void UpdateOutput() 
        {
            foreach (var layer in _layers)
                layer.UpdateResult();
        }
        public void UpdateWeights(double lr) 
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].Train(lr);
            }
        }
        public double UpdateErrors(double[] target) 
        {
            double err = 0;
            for (int i = 0; i < lastErrors.Length; i++) 
            {
                lastErrors[i] = target[i] - Output[i];
                err += Math.Abs(lastErrors[i]);
            }
            err /= lastErrors.Length;
            for (int i = _layers.Length - 1; i >= 1; i--) 
            {
                _layers[i].UpdateErrors(_layers[i - 1].Errors);
            }
            return err;
        }
        public void Train(IEnumerable<Tuple<double[],double[]>> dataset,double rate, double targetAccuracy, int accuracyIterations = 100, int maxIterations=-1) 
        {
            int iterations = 0;

            var enumerator = dataset.GetEnumerator();
            var errors = new double[accuracyIterations];
            for (int i = 0; i < accuracyIterations; i++)
                errors[i] = 1;
            int errcounter = 0;
            while (true) 
            {
                iterations++;
                if (!enumerator.MoveNext())
                {
                    enumerator = dataset.GetEnumerator();
                    if (!enumerator.MoveNext())
                        throw new Exception();
                }
                var example = enumerator.Current;

                example.Item1.CopyTo(Input, 0);
                UpdateOutput();
                var res = UpdateErrors(example.Item2);
                errors[errcounter] = res;
                errcounter = (errcounter + 1) % accuracyIterations;
                if (errcounter == 0) 
                {
                    res = 0;
                    for (int i = 0; i < accuracyIterations; i++) 
                    {
                        res += errors[i];
                    }
                    res /= accuracyIterations;
                    Console.WriteLine(res);
                    if (res <= targetAccuracy)
                    {
                        break;
                    }
                }
                UpdateWeights(rate);
            }
        }
        public void RandomizeWeights(double range) 
        {
            foreach (var l in _layers)
                l.RandomizeWeights(range);
        }
    }
}

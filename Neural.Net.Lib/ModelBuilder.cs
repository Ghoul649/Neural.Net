using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class ModelBuilder
    {
        private int inputSize;
        private List<LayerInfo> layerInfos = new List<LayerInfo>();
        public ModelBuilder(int inputSize) 
        {
            this.inputSize = inputSize;
        }
        public void AddFullConnection(int size, ActivateFunction activateFunction = null) 
        {
            layerInfos.Add(new LayerInfo() { Size = size, ActivateFunction = activateFunction ?? ActivateFunction.Logistic, LayerType = LayerType.FullConnection });
        }
        public Model Build() 
        {
            ILayer[] layers = new ILayer[layerInfos.Count];
            int i = 0;
            var input = new double[inputSize];
            foreach (var info in layerInfos)
            {
                if (info.LayerType == LayerType.FullConnection)
                {
                    ILayer layer = new FullConnection(input, info.Size, info.ActivateFunction);
                    input = layer.Output;
                    layers[i++] = layer; 
                    continue;
                }
            }
            return new Model(layers);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    enum LayerType
    {
        FullConnection
    }
    class LayerInfo
    {
        public LayerType LayerType { get; set; }
        public int Size { get; set; }
        public ActivateFunction ActivateFunction { get; set; }
    }
}

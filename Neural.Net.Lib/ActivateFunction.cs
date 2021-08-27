using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetLib
{
    public class ActivateFunction
    {
        public delegate double FunctionDelegate(double x);
        public FunctionDelegate F { get; private set; }
        public FunctionDelegate DF { get; private set; }

        public static ActivateFunction Logistic { get; } = new ActivateFunction() 
        { 
            F = x => 1 / (1 + Math.Pow(Math.E, -x)) ,
            DF = x => 
            {
                double a = 1 / (1 + Math.Pow(Math.E, -x));
                return a * (1 - a);
            }
        };
        public static ActivateFunction Linear { get; } = new ActivateFunction()
        {
            F = x => x,
            DF = x => 1
        };
    }
}

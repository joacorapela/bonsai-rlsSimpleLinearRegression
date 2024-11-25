
using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class RLSestimator
{
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] w0 { get; set; }

    [XmlIgnore]
    [TypeConverter(typeof(MultidimensionalArrayConverter))]
    public double[,] P0 { get; set; }

    [Browsable(false)]
    [XmlElement("S0")]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string KernelXml
    {
        get { return ArrayConvert.ToString(P0, CultureInfo.InvariantCulture); }
        set { P0 = (double[,])ArrayConvert.ToArray(value, 2, typeof(double), CultureInfo.InvariantCulture); }
    }

    public double lambda { get; set; }

    public IObservable<RLSdataItem> Process(IObservable<RegressionObservation> observations)
    {
        Console.WriteLine("RLSestimator Process called");
        return observations.Scan(
            new RLSdataItem
            {
                w = Vector<double>.Build.DenseOfArray(w0),
                P = Matrix<double>.Build.DenseOfArray(P0)
            },
            (state, obs) =>
            {
                var updateRes = RecursiveLeastSquares.Update(state.w, state.P, obs.t, obs.phi, lambda);
                RLSdataItem rlsDI = new RLSdataItem { w = updateRes.Item1, P = updateRes.Item2 };
                return rlsDI;
            });
    }
}

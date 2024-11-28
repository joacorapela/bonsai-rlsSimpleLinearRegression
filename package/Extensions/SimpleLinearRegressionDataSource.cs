using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;
using JoacoRapela.Bonsai.ML.RecursiveLeastSquares;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SimpleLinearRegressionDataSource{
    public double a0 { get; set; }
    public double a1 { get; set; }
    public double sigma { get; set; }

    public IObservable<RegressionObservation> Process(IObservable<long> source)
    {
        Console.WriteLine("RegressionObservationsDataSource::Process called");
        System.Random rng = SystemRandomSource.Default;

        return source.Select( input =>
        {
            Console.WriteLine("RegressionObservationsDataSource::Process generating a new observation");
            double x = 4 * rng.NextDouble() - 2.0;
            double epsilon = Normal.Sample(0.0, this.sigma);
            double y = this.a0 + this.a1 * x;
            double t = y + epsilon;

            double[] aux = new[] { 1, x };
            Vector<double> phi = Vector<double>.Build.DenseOfArray(aux);
            RegressionObservation observation = new RegressionObservation();
            observation.phi = phi;
            observation.t = t;

            return observation;
        });
    }
}

using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Drawing;
using Bonsai.Design;
using System.Windows.Forms;
using Bonsai.Design.Visualizers;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using ScottPlot;
using ScottPlot.Plottable;

[assembly: TypeVisualizer(typeof(BatchRegressionObsAndPredictionsVisualizer), Target=typeof(Tuple<IList<RegressionObservation>, RLSdataItem>))] 

public class BatchRegressionObsAndPredictionsVisualizer : DialogTypeVisualizer
{
    private static ScottPlot.FormsPlot _formsPlot1;

    public override void Load(IServiceProvider provider)
    {
        _formsPlot1 = new ScottPlot.FormsPlot() { Dock = DockStyle.Fill };
        _formsPlot1.Plot.YLabel("f(x)");
        _formsPlot1.Plot.XLabel("x");

        var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
        if (visualizerService != null)
        {
            visualizerService.AddControl(_formsPlot1);
        }
    }

    public override void Show(object aValue)
    {
        Console.WriteLine("BatchRegressionObsAndPredictionsVis::Show called");

        // Step 1: Cast `aValue` to the tuple type
        var tuple = (Tuple<IList<RegressionObservation>, RLSdataItem>)aValue;

        // Step 2: Deconstruct the tuple
        var batchRObs = tuple.Item1;
        var rlsDI = tuple.Item2;

        // computer predictions
        double[] x = new double[batchRObs.Count];
        double[] t = new double[batchRObs.Count];
        for (int i=0; i<batchRObs.Count; i++)
        {
            x[i] = batchRObs[i].phi[1];
            t[i] = batchRObs[i].t;
        }

        double xMin = -2.0;
        double xMax = 2.0;

        int nDense = 100;
        double step = (xMax - xMin) / nDense;
        var xDense = Enumerable.Range(0, (int)Math.Ceiling((xMax - xMin) / step))
            .Select(i => xMin + i * step).ToArray();
        double[] mean = new double[xDense.Length];
        for (int i=0; i<xDense.Length; i++)
        {
            double[] aux = {1.0, xDense[i]};
            Vector<double> u = Vector<double>.Build.DenseOfArray(aux);
            mean[i] = RecursiveLeastSquares.Predict(rlsDI.w, u);
        }

        // plot means and 95% ci for xDense
        _formsPlot1.Plot.Clear();

        _formsPlot1.Plot.AddScatter(xDense, mean, Color.Blue, label: "Predictions");

        // plot data
        _formsPlot1.Plot.AddScatter(x, t, Color.Red, lineWidth: 0, label: "Observations");
        _formsPlot1.Plot.YLabel("f(x)");
        _formsPlot1.Plot.XLabel("x");
        var legend = _formsPlot1.Plot.Legend();
        legend.Location = Alignment.UpperLeft;


        _formsPlot1.Refresh();
    }

    public override void Unload()
    {
    }
}

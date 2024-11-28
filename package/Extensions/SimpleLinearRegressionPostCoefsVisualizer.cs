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
using JoacoRapela.Bonsai.ML.RecursiveLeastSquares;

[assembly: TypeVisualizer(typeof(SimpleLinearRegressionPostCoefsVisualizer), Target=typeof(RLSestimator))] 

public class SimpleLinearRegressionPostCoefsVisualizer : DialogTypeVisualizer
{
    private static ScottPlot.FormsPlot _formsPlot1;
    private static Heatmap _hm;
    private static double[,] _buffer;
    private static double[] _x;
    private static double[] _y;

    public override void Load(IServiceProvider provider)
    {
        _formsPlot1 = new ScottPlot.FormsPlot() { Dock = DockStyle.Fill };
        _x = MathNet.Numerics.Generate.LinearRange(-1.0, 0.01, 1.0);
        _y = MathNet.Numerics.Generate.LinearRange(-1.0, 0.01, 1.0);
        _buffer = new double[_x.Length, _y.Length];

        // Add sample data to the plot
        _hm = _formsPlot1.Plot.AddHeatmap(_buffer, lockScales: false);
	_formsPlot1.Plot.XLabel("b0");
	_formsPlot1.Plot.YLabel("b1");
        _hm.FlipVertically = true;
        _hm.XMin = -1.0;
        _hm.XMax = 1.0;
        _hm.YMin = -1.0;
        _hm.YMax = 1.0;
        // _formsPlot1.Frameless();
        _formsPlot1.Refresh();

        var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
        if (visualizerService != null)
        {
            visualizerService.AddControl(_formsPlot1);
        }
    }

    public override void Show(object value)
    {
	RLSdataItem rlsDataItem = (RLSdataItem) value;
        double[,] smallCov = {{0.001,0}, {0,0.001}};
        computeMultivariateGaussianPDForGrid(_buffer, rlsDataItem.w.ToArray(), smallCov);
        _hm.Update(_buffer);
        _formsPlot1.Refresh();
    }

    public override void Unload()
    {
    }

    private static void computeMultivariateGaussianPDForGrid(double[,] buffer, double[] mn, double[,] Sn)
    {
        Vector<double> mnVec = Vector<double>.Build.DenseOfArray(mn);
        Matrix<double> SnMat = Matrix<double>.Build.DenseOfArray(Sn);
        double[] eval_loc_buffer = new double[2];
        MatrixNormal matrixNormal = new MatrixNormal(mnVec.ToColumnMatrix(), SnMat, Matrix<double>.Build.DenseIdentity(1));
        for (int i = 0; i < _x.Length; i++)
        {
            eval_loc_buffer[0] = _x[i];
            for (int j = 0; j < _y.Length; j++)
            {
                eval_loc_buffer[1] = _y[j];
                Vector<double> eval_loc = Vector<double>.Build.Dense(eval_loc_buffer);
                buffer[j, i] = matrixNormal.Density(eval_loc.ToColumnMatrix());
            }
        }
    }
}

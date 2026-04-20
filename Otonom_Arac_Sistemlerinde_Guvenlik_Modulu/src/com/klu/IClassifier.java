package com.klu;

import java.util.List;

public interface IClassifier {
    void train(List<DataPoint> dataset, int epochs, double learningRate, double lambda);
    int predict(double[] coordinates);
}

package com.klu;

import java.util.List;

public class LinearSVM implements IClassifier {
    private double[] weights;
    private double bias;

    public LinearSVM(int featureCount) {
        // Ağırlıkları sıfır bellek sızıntısı prensibiyle sadece bir kez oluşturuyoruz.
        this.weights = new double[featureCount];
        this.bias = 0.0;
    }

    @Override
    public void train(List<DataPoint> dataset, int epochs, double learningRate, double lambda) {
        // Zaman Karmaşıklığı (Time Complexity): O(Epochs * N)
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (DataPoint point : dataset) {
                double[] x = point.getCoordinates();
                int y = point.getLabel();

                // y_i * (w * x_i + b) >= 1 durumu kontrol ediliyor.
                double margin = y * (dotProduct(weights, x) + bias);

                if (margin >= 1) {
                    // Nokta güvenlik koridorunun dışındaysa sadece ağırlıkları küçült (Regularization)
                    for (int i = 0; i < weights.length; i++) {
                        weights[i] -= learningRate * (2 * lambda * weights[i]);
                    }
                } else {
                    // Nokta koridor ihlali yapıyorsa, ağırlıkları ve bias'ı güncelle
                    for (int i = 0; i < weights.length; i++) {
                        weights[i] -= learningRate * (2 * lambda * weights[i] - y * x[i]);
                    }
                    bias -= learningRate * (-y); // Bias güncellemesi
                }
            }
        }
    }

    @Override
    public int predict(double[] coordinates) {
        double result = dotProduct(weights, coordinates) + bias;
        return result >= 0 ? 1 : -1;
    }

    // İki vektörün nokta çarpımını (Dot Product) hesaplayan yardımcı fonksiyon
    private double dotProduct(double[] w, double[] x) {
        double sum = 0.0;
        for (int i = 0; i < w.length; i++) {
            sum += w[i] * x[i];
        }
        return sum;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}

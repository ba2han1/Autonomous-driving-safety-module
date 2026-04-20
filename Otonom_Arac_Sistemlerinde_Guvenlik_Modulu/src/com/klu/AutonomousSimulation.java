package com.klu;

import java.util.ArrayList;
import java.util.List;

public class AutonomousSimulation {
    public static void main(String[] args) {
        System.out.println("Otonom Araç Güvenlik Modülü Simülasyonu Başlıyor...\n");

        List<DataPoint> dataset = new ArrayList<>();

        // Sınıf 1 Engelleri (-1)
        dataset.add(new DataPoint(2, 8, -1));
        dataset.add(new DataPoint(3, 9, -1));
        dataset.add(new DataPoint(4, 8, -1));

        // Sınıf 2 Engelleri (+1)
        dataset.add(new DataPoint(2, 3, 1));
        dataset.add(new DataPoint(3, 2, 1));
        dataset.add(new DataPoint(4, 3, 1));

        IClassifier svm = new LinearSVM(2);

        // --- KRONOMETRE BAŞLANGICI ---
        long startTime = System.nanoTime();

        // Hiperparametreler
        svm.train(dataset, 1000, 0.001, 0.01);

        // --- KRONOMETRE BİTİŞİ ---
        long endTime = System.nanoTime();

        // Nanosaniyeyi milisaniyeye çeviriyoruz (1 Milisaniye = 1.000.000 Nanosaniye)
        double executionTimeMs = (endTime - startTime) / 1_000_000.0;

        LinearSVM trainedModel = (LinearSVM) svm;
        double[] finalWeights = trainedModel.getWeights();
        double finalBias = trainedModel.getBias();

        System.out.println("--- EĞİTİM SONUÇLARI ---");
        System.out.printf("Ayrıştırıcı Denklem: %.4fx + %.4fy + %.4f = 0\n",
                finalWeights[0], finalWeights[1], finalBias);

        // Süreyi konsola yazdırıyoruz
        System.out.printf("Algoritma Çalışma Süresi: %.4f ms\n\n", executionTimeMs);

        // Test aşaması
        double[] testPoint = {3, 5.5};
        int prediction = svm.predict(testPoint);
        System.out.println("Test Noktası (3, 5.5) tahmini: Sınıf " + prediction);
    }
}
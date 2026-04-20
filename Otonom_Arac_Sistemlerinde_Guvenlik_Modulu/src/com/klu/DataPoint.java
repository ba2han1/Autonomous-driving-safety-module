package com.klu;

public class DataPoint {
    private final double[] coordinates;
    private final int label; // Sınıf etiketi: +1 veya -1

    public DataPoint(double x, double y, int label) {
        this.coordinates = new double[]{x, y};
        this.label = label;
    }

    public double[] getCoordinates() {
        return coordinates;
    }

    public int getLabel() {
        return label;
    }
}

package old;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.MLResettable;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.temporal.TemporalDataDescription;
import org.encog.ml.data.temporal.TemporalMLDataSet;
import org.encog.ml.data.temporal.TemporalPoint;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.RequiredImprovementStrategy;
import org.encog.ml.train.strategy.end.SimpleEarlyStoppingStrategy;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class MLPNoValidation {

    private final int INPUT_WINDOW_SIZE;

    private final int HIDDEN_LAYER_NEURONS;

    private final int PREDICT_WINDOW_SIZE;

    private final double TRAIN_TO_ERROR;

    private final int VARIABLES;

    private final boolean INPUT;

    private final ArrayList<NormalizedField> NORMALIZATIONS;

    public MLPNoValidation(boolean INPUT, int VARIABLES, int INPUT_WINDOW_SIZE, int HIDDEN_LAYER_NEURONS, int PREDICT_WINDOW_SIZE, double TRAIN_TO_ERROR, ArrayList<NormalizedField> NORM) {
        this.INPUT_WINDOW_SIZE = INPUT_WINDOW_SIZE;
        this.HIDDEN_LAYER_NEURONS = HIDDEN_LAYER_NEURONS;
        this.PREDICT_WINDOW_SIZE = PREDICT_WINDOW_SIZE;
        this.TRAIN_TO_ERROR = TRAIN_TO_ERROR;
        this.VARIABLES = VARIABLES;
        this.INPUT = INPUT;
        this.NORMALIZATIONS = NORM;
    }
    
    public double execute(int numeroExecucao, Date date) throws IOException {

        TemporalMLDataSet trainingData = createTraining(new File(new File(".."), "25_treinamento.csv"));
        TemporalMLDataSet validatingData = createTraining(new File(new File(".."), "25_teste.csv"));
        
        MLRegression model = trainModel(
                trainingData,
                validatingData,
                MLMethodFactory.TYPE_FEEDFORWARD,
                "?:B->SIGMOID->" + HIDDEN_LAYER_NEURONS + ":B->SIGMOID->?",
                MLTrainFactory.TYPE_RPROP,
                "");

        double error = predict(new File(new File(".."), "25_teste.csv"), model, numeroExecucao, date);

        Encog.getInstance().shutdown();

        return error;
    }
    
    public TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');

        for (int x = 0; csv.next(); x++) {

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);

            for (int y = 0; y < VARIABLES; y++) {

                point.setData(y, NORMALIZATIONS.get(y).normalize(csv.getDouble(y)));

            }

            trainingData.getPoints().add(point);

        }
        csv.close();

        trainingData.generate();
        return trainingData;
    }

    public TemporalMLDataSet initDataSet() {
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);
        dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, INPUT, true));
        for (int x = 0; x < VARIABLES; x++) {
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
        }
        return dataSet;
    }

    public MLRegression trainModel(
            MLDataSet trainingData,
            MLDataSet validatingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs) {

        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(methodName, methodArchitecture, trainingData.getInputSize(), trainingData.getIdealSize());
        
        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method, trainingData, trainerName, trainerArgs);

        train.addStrategy(new SimpleEarlyStoppingStrategy(validatingData));
        
        //if (method instanceof MLResettable && !(train instanceof ManhattanPropagation)) {
        //    train.addStrategy(new RequiredImprovementStrategy(500));
        //}

        EncogUtility.trainToError(train, TRAIN_TO_ERROR);

        return (MLRegression) train.getMethod();
    }

    public double predict(File rawFile, MLRegression model, int numeroExecucao, Date date) throws IOException {
        
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');

        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        new File(new File(".."), "\\MLP_" + VARIABLES + "\\" + dateFormat.format(date)).mkdirs();
        FileWriter arq = new FileWriter(new File(new File(".."), "\\MLP_" + VARIABLES + "\\" + dateFormat.format(date) + "\\MLP_" + VARIABLES + "_" + dateFormat.format(date) + "_" + numeroExecucao + ".csv"));
        PrintWriter gravarArq = new PrintWriter(arq);
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);
        double soma = 0;
        int x;
        gravarArq.print("Dia;Real;Previsto;Erro Relativo Percentual\n");

        for (x = 0; csv.next(); x++) {

            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {

                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double MP = NORMALIZATIONS.get(0).deNormalize(modelOutput.getData(0));

                gravarArq.print(x + ";" + nf.format(csv.getDouble(0)) + ";" + nf.format(MP) + ";" + nf.format(Math.abs(((MP - csv.getDouble(0)) / csv.getDouble(0)) * 100)) + "\n");
                soma = soma + Math.abs(((MP - csv.getDouble(0)) / csv.getDouble(0)) * 100);

                trainingData.getPoints().remove(0);
            }

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);
            for (int y = 0; y < VARIABLES; y++) {

                point.setData(y, NORMALIZATIONS.get(y).normalize(csv.getDouble(y)));

            }
            trainingData.getPoints().add(point);
        }
        arq.close();
        csv.close();

        //System.out.println("Média Erro do " + dateFormat.format(date) + " = " + nf.format(soma / n));
        trainingData.generate();
        return soma / x;
    }

    public static void main(String[] args) throws IOException {
        Date date = new Date();
        boolean input = true;
        int variables = 4;
        int repetitions = 10;
        int inputWindowSize = 4;
        int hiddenLayerNeurons = 22;
        int predictWindowSize = 1;
        double trainToError = 0.002;
        ArrayList<Double> results = new ArrayList<>();

        ArrayList<NormalizedField> normalizations = new ArrayList<>();

        normalizations.add(new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0));
        normalizations.add(new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0));
        normalizations.add(new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0));
        normalizations.add(new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0));

        for (int x = 1; x <= repetitions; x++) {
            results.add(new MLPNoValidation(input,
                    variables,
                    inputWindowSize,
                    hiddenLayerNeurons,
                    predictWindowSize,
                    trainToError,
                    normalizations).execute(x, date));
        }

        System.out.println(new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss").format(date));
        System.out.println("Entradas\t" + inputWindowSize);
        System.out.println("Camadas Oculta\t" + hiddenLayerNeurons);
        System.out.println("Janela\t" + predictWindowSize);
        System.out.println("Erro\t" + trainToError);

        double soma = 0;

        for (int x = 0; x < repetitions; x++) {
            System.out.println(x + 1 + "\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(results.get(x)));
            soma = soma + results.get(x);
        }

        System.out.println("Media\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(soma / 10));
    }

}

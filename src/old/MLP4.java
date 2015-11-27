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

public class MLP4 {

    private static final File MYDIR = new File("..");

    private static int INPUT_WINDOW_SIZE;

    private static int CAMADA_OCULTA;

    private static int PREDICT_WINDOW_SIZE;

    private static double TRAIN_TO_ERROR;

    private static final NormalizedField normMP = new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0);
    private static final NormalizedField normTEMP = new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0);
    private static final NormalizedField normUR = new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0);
    private static final NormalizedField normVV = new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0);

    public MLP4(int INPUT_WINDOW_SIZE, int CAMADA_OCULTA, int PREDICT_WINDOW_SIZE, double TRAIN_TO_ERROR) {

        MLP4.INPUT_WINDOW_SIZE = INPUT_WINDOW_SIZE;
        MLP4.CAMADA_OCULTA = CAMADA_OCULTA;
        MLP4.PREDICT_WINDOW_SIZE = PREDICT_WINDOW_SIZE;
        MLP4.TRAIN_TO_ERROR = TRAIN_TO_ERROR;

    }

    public static TemporalMLDataSet initDataSet() {
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);

        TemporalDataDescription MP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true);

        TemporalDataDescription TEMP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription UR = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription VV = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        dataSet.addDescription(MP);
        dataSet.addDescription(TEMP);
        dataSet.addDescription(UR);
        dataSet.addDescription(VV);
        return dataSet;
    }

    public static MLRegression trainModel(
            MLDataSet trainingData,
            MLDataSet validadingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs) {

        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(methodName, methodArchitecture, trainingData.getInputSize(), trainingData.getIdealSize());

        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method, trainingData, trainerName, trainerArgs);

        SimpleEarlyStoppingStrategy stop = new SimpleEarlyStoppingStrategy(validadingData, 100);

        train.addStrategy(stop);

        //if (method instanceof MLResettable && !(train instanceof ManhattanPropagation)) {
        //    train.addStrategy(new RequiredImprovementStrategy(500));
        //}
        int epoch = 1;

        while (!stop.shouldStop()) {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Validation Error: " + stop.getValidationError());
            epoch++;
        }

        train.finishTraining();

        return (MLRegression) train.getMethod();
    }

    public static TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');
        int x = 0;
        while (csv.next()) {
            double MP = csv.getDouble(0);
            double TEMP = csv.getDouble(1);
            double UR = csv.getDouble(2);
            double VV = csv.getDouble(3);

            int sequenceNumber = x;

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normMP.normalize(MP));
            point.setData(1, normTEMP.normalize(TEMP));
            point.setData(2, normUR.normalize(UR));
            point.setData(3, normVV.normalize(VV));
            trainingData.getPoints().add(point);
            x++;
        }
        csv.close();

        trainingData.generate();
        return trainingData;
    }

    public static double predict(File rawFile, MLRegression model, int numeroExecucao, Date date) throws IOException {

        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');
        int x = 0;
        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        new File(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date)).mkdirs();
        FileWriter arq = new FileWriter(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date) + "\\MLP4_" + dateFormat.format(date) + "_" + numeroExecucao + ".csv");
        PrintWriter gravarArq = new PrintWriter(arq);
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);
        double soma = 0;
        int n = 0;

        gravarArq.print("Dia;Real;Previsto;Erro Relativo Percentual\n");

        while (csv.next()) {
            double MP = csv.getDouble(0);
            double TEMP = csv.getDouble(1);
            double UR = csv.getDouble(2);
            double VV = csv.getDouble(3);

            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {

                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double mp = normMP.deNormalize(modelOutput.getData(0));

                gravarArq.print(x + ";" + nf.format(MP) + ";" + nf.format(mp) + ";" + nf.format(Math.abs(((mp - MP) / MP) * 100)) + "\n");
                soma = soma + Math.abs(((mp - MP) / MP) * 100);
                n++;

                trainingData.getPoints().remove(0);
            }

            int sequenceNumber = x;

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normMP.normalize(MP));
            point.setData(1, normTEMP.normalize(TEMP));
            point.setData(2, normUR.normalize(UR));
            point.setData(3, normVV.normalize(VV));
            trainingData.getPoints().add(point);
            x++;
        }
        arq.close();
        csv.close();

        //System.out.println("Média Erro do " + dateFormat.format(date) + " = " + nf.format(soma / n));
        trainingData.generate();
        return soma / n;
    }

    public double executar(int numeroExecucao, Date date) throws IOException {
        File arquivoTeste = new File(MYDIR, "25_teste.csv");
        File arquivoTreinamento = new File(MYDIR, "25_treinamento2.csv");
        File arquivoValidacao = new File(MYDIR, "25_validacao2.csv");


        TemporalMLDataSet trainingData = createTraining(arquivoTreinamento);
        TemporalMLDataSet validadingData = createTraining(arquivoValidacao);

        MLRegression model = trainModel(
                trainingData,
                validadingData,
                MLMethodFactory.TYPE_FEEDFORWARD,
                "?:B->SIGMOID->" + CAMADA_OCULTA + ":B->SIGMOID->?",
                MLTrainFactory.TYPE_RPROP,
                "");

        double erro = predict(arquivoTeste, model, numeroExecucao, date);

        Encog.getInstance().shutdown();

        return erro;
    }

    public static void main(String[] args) throws IOException {
        Date date = new Date();
        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");

        int ENTRADA = 4;
        int OCULTA = 22;
        int PREVER = 1;
        double ERRO = 0.002;//0.0036;

        ArrayList<Double> resultados = new ArrayList<>();

        MLP4 execucao1 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao1.executar(1, date));
        MLP4 execucao2 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao2.executar(2, date));
        MLP4 execucao3 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao3.executar(3, date));
        MLP4 execucao4 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao4.executar(4, date));
        MLP4 execucao5 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao5.executar(5, date));
        MLP4 execucao6 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao6.executar(6, date));
        MLP4 execucao7 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao7.executar(7, date));
        MLP4 execucao8 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao8.executar(8, date));
        MLP4 execucao9 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao9.executar(9, date));
        MLP4 execucao10 = new MLP4(ENTRADA, OCULTA, PREVER, ERRO);
        resultados.add(execucao10.executar(10, date));

        FileWriter arq = new FileWriter(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date) + "\\MLP4_" + dateFormat.format(date) + "_INFO.txt");
        PrintWriter gravarArq = new PrintWriter(arq);

        System.out.println(dateFormat.format(date));
        gravarArq.println(dateFormat.format(date));

        gravarArq.println("Entradas\t" + ENTRADA);
        gravarArq.println("Camadas Oculta\t" + OCULTA);
        gravarArq.println("Janela\t" + PREVER);
        gravarArq.println("Erro\t" + ERRO);
        System.out.println("Entradas\t" + ENTRADA);
        System.out.println("Camadas Oculta\t" + OCULTA);
        System.out.println("Janela\t" + PREVER);
        System.out.println("Erro\t" + ERRO);

        double soma = 0;
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);

        for (int x = 0; x < 10; x++) {
            System.out.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            gravarArq.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            soma = soma + resultados.get(x);
        }
        System.out.println("Media\t" + nf.format(soma / 10));
        gravarArq.println("Media\t" + nf.format(soma / 10));

        arq.close();
    }

}
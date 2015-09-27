
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
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class MLP {

    private final int INPUT_WINDOW_SIZE;

    private final int CAMADA_OCULTA;

    private final int PREDICT_WINDOW_SIZE;

    private final double TRAIN_TO_ERROR;
    
    private final double VARIAVEIS;

    private final ArrayList<NormalizedField> NORM = new ArrayList<>();

    public MLP(int VARIAVEIS, int INPUT_WINDOW_SIZE, int CAMADA_OCULTA, int PREDICT_WINDOW_SIZE, double TRAIN_TO_ERROR) {

        this.INPUT_WINDOW_SIZE = INPUT_WINDOW_SIZE;
        this.CAMADA_OCULTA = CAMADA_OCULTA;
        this.PREDICT_WINDOW_SIZE = PREDICT_WINDOW_SIZE;
        this.TRAIN_TO_ERROR = TRAIN_TO_ERROR;
        this.VARIAVEIS = VARIAVEIS;
        NORM.add(new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0));
        NORM.add(new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0));
        NORM.add(new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0));
        NORM.add(new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0));

    }

    public TemporalMLDataSet initDataSet() {
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);
        if (VARIAVEIS == 1) {
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true));
        } else if (VARIAVEIS == 3) {
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, false, true));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
        } else if (VARIAVEIS == 4) {
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
            dataSet.addDescription(new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false));
        }

        return dataSet;
    }

    public MLRegression trainModel(
            MLDataSet trainingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs) {

        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(methodName, methodArchitecture, trainingData.getInputSize(), trainingData.getIdealSize());

        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method, trainingData, trainerName, trainerArgs);

        if (method instanceof MLResettable && !(train instanceof ManhattanPropagation)) {
            train.addStrategy(new RequiredImprovementStrategy(500));
        }

        EncogUtility.trainToError(train, TRAIN_TO_ERROR);

        return (MLRegression) train.getMethod();
    }

    public TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');

        for (int x = 0; csv.next(); x++) {

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);

            for (int y = 0; y < VARIAVEIS; y++) {

                point.setData(y, NORM.get(y).normalize(csv.getDouble(y)));

            }

            trainingData.getPoints().add(point);

        }
        csv.close();

        trainingData.generate();
        return trainingData;
    }

    public double predict(File rawFile, MLRegression model, int numeroExecucao, Date date) throws IOException {

        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');

        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        new File(System.getProperty("user.dir") + "\\MLP_" + VARIAVEIS + "\\" + dateFormat.format(date)).mkdirs();
        FileWriter arq = new FileWriter(System.getProperty("user.dir") + "\\MLP_" + VARIAVEIS + "\\" + dateFormat.format(date) + "\\MLP_" + VARIAVEIS + "_" + dateFormat.format(date) + "_" + numeroExecucao + ".csv");
        PrintWriter gravarArq = new PrintWriter(arq);
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);
        double soma = 0;
        int x;
        gravarArq.print("Dia;Real;Previsto;Erro Relativo Percentual\n");

        for (x = 0; csv.next(); x++) {

            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {

                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double MP = NORM.get(0).deNormalize(modelOutput.getData(0));

                gravarArq.print(x + ";" + nf.format(csv.getDouble(0)) + ";" + nf.format(MP) + ";" + nf.format(Math.abs(((MP - csv.getDouble(0)) / csv.getDouble(0)) * 100)) + "\n");
                soma = soma + Math.abs(((MP - csv.getDouble(0)) / csv.getDouble(0)) * 100);

                trainingData.getPoints().remove(0);
            }

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);
            for (int y = 0; y < VARIAVEIS; y++) {

                point.setData(y, NORM.get(y).normalize(csv.getDouble(y)));

            }
            trainingData.getPoints().add(point);
        }
        arq.close();
        csv.close();

        //System.out.println("Média Erro do " + dateFormat.format(date) + " = " + nf.format(soma / n));
        trainingData.generate();
        return soma / x;
    }

    public double executar(int numeroExecucao, Date date) throws IOException {
        File arquivoTeste = new File(new File(".."), "25_teste.csv");
        File arquivoTreinamento = new File(new File(".."), "25_treinamento.csv");

        TemporalMLDataSet trainingData = createTraining(arquivoTreinamento);

        MLRegression model = trainModel(
                trainingData,
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
        
        int variaveis = 1;
        int janelaEntrada = 4;
        int oculta = 22;
        int janelaPrever = 1;
        double ERRO = 0.002;

        ArrayList<Double> resultados = new ArrayList<>();

        MLP execucao1 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao1.executar(1, date));
        MLP execucao2 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao2.executar(2, date));
        MLP execucao3 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao3.executar(3, date));
        MLP execucao4 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao4.executar(4, date));
        MLP execucao5 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao5.executar(5, date));
        MLP execucao6 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao6.executar(6, date));
        MLP execucao7 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao7.executar(7, date));
        MLP execucao8 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao8.executar(8, date));
        MLP execucao9 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao9.executar(9, date));
        MLP execucao10 = new MLP(variaveis, janelaEntrada, oculta, janelaPrever, ERRO);
        resultados.add(execucao10.executar(10, date));

        //FileWriter arq = new FileWriter(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date) + "\\MLP4_" + dateFormat.format(date) + "_INFO.txt");
        //PrintWriter gravarArq = new PrintWriter(arq);

        System.out.println(dateFormat.format(date));
        //gravarArq.println(dateFormat.format(date));

        //gravarArq.println("Entradas\t" + janelaEntrada);
        //gravarArq.println("Camadas Oculta\t" + oculta);
        //gravarArq.println("Janela\t" + janelaPrever);
        //gravarArq.println("Erro\t" + ERRO);
        System.out.println("Entradas\t" + janelaEntrada);
        System.out.println("Camadas Oculta\t" + oculta);
        System.out.println("Janela\t" + janelaPrever);
        System.out.println("Erro\t" + ERRO);

        double soma = 0;
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);

        for (int x = 0; x < 10; x++) {
            System.out.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            //gravarArq.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            soma = soma + resultados.get(x);
        }
        System.out.println("Media\t" + nf.format(soma / 10));
        //gravarArq.println("Media\t" + nf.format(soma / 10));

        //arq.close();
    }

}


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

    private final File MYDIR = new File("..");
    private final int INPUT_WINDOW_SIZE;
    private final int CAMADA_OCULTA;
    private final int PREDICT_WINDOW_SIZE;
    private final double TRAIN_TO_ERROR;
    private final int INPUTS;
    private final ArrayList<NormalizedField> NORM;

    public MLP(int INPUTS, int INPUT_WINDOW_SIZE, int CAMADA_OCULTA, int PREDICT_WINDOW_SIZE, double TRAIN_TO_ERROR) {

        this.INPUT_WINDOW_SIZE = INPUT_WINDOW_SIZE;
        this.CAMADA_OCULTA = CAMADA_OCULTA;
        this.PREDICT_WINDOW_SIZE = PREDICT_WINDOW_SIZE;
        this.TRAIN_TO_ERROR = TRAIN_TO_ERROR;
        this.INPUTS = INPUTS;
        NORM = new ArrayList<>();

        if (INPUTS == 4) {

            NORM.add(new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0));
            NORM.add(new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0));
            NORM.add(new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0));
            NORM.add(new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0));

        } else if (INPUTS == 3) {

            NORM.add(new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0));
            NORM.add(new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0));
            NORM.add(new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0));

        } else if (INPUTS == 1) {

            NORM.add(new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0));

        }
    }

    public TemporalMLDataSet initDataSet() {
        TemporalMLDataSet dataSet;
        dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);

        TemporalDataDescription MP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true);
        dataSet.addDescription(MP);

        TemporalDataDescription TEMP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription UR = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription VV = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        dataSet.addDescription(TEMP);
        dataSet.addDescription(UR);
        dataSet.addDescription(VV);

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
        int x = 0;
        ArrayList<Double> CSV = new ArrayList<>();
        while (csv.next()) {

            for (int y = 0; y < INPUTS; y++) {
                CSV.add(csv.getDouble(y));
            }

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);
            for (int y = 0; y < INPUTS; y++) {
                point.setData(y, NORM.get(y).normalize(CSV.get(y)));
            }
            trainingData.getPoints().add(point);

            x++;
        }
        csv.close();

        trainingData.generate();
        return trainingData;
    }

    public double predict(File rawFile, MLRegression model, int numeroExecucao, Date date) throws IOException {

        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');
        int x = 0;
        //DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        //new File(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date)).mkdirs();
        //FileWriter arq = new FileWriter(System.getProperty("user.dir") + "\\MLP4\\" + dateFormat.format(date) + "\\MLP4_" + dateFormat.format(date) + "_" + numeroExecucao + ".csv");
        //PrintWriter gravarArq = new PrintWriter(arq);
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);
        double soma = 0;
        int n = 0;

        //gravarArq.print("Dia;Real;Previsto;Erro Relativo Percentual\n");
        ArrayList<Double> CSV = new ArrayList<>();
        while (csv.next()) {
            for (int y = 0; y < INPUTS; y++) {
                CSV.add(csv.getDouble(y));
            }

            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {

                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double mp = NORM.get(0).deNormalize(modelOutput.getData(0));
                double MP = CSV.get(0);
                //gravarArq.print(x + ";" + nf.format(MP) + ";" + nf.format(mp) + ";" + nf.format(Math.abs(((mp - MP) / MP) * 100)) + "\n");
                soma = soma + Math.abs(((mp - MP) / MP) * 100);
                n++;

                trainingData.getPoints().remove(0);
            }

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(x);
            for (int y = 0; y < INPUTS; y++) {
                point.setData(y, NORM.get(y).normalize(CSV.get(y)));
            }
            trainingData.getPoints().add(point);

            x++;
        }
        //arq.close();
        csv.close();

        //System.out.println("Média Erro do " + dateFormat.format(date) + " = " + nf.format(soma / n));
        trainingData.generate();
        return soma / n;
    }

    public double executar(int numeroExecucao, Date date) throws IOException {
        File arquivoTeste = new File(MYDIR, "25_teste.csv");
        File arquivoTreinamento = new File(MYDIR, "25_treinamento.csv");

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
        //DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");

        int ENTRADA = 4;
        int OCULTA = 22;
        int PREVER = 1;
        double ERRO = 0.0001;

        ArrayList<Double> resultados = new ArrayList<>();

        //for (int x = 0; x < 10; x++) {
            MLP execucao = new MLP(4, ENTRADA, OCULTA, PREVER, ERRO);
            resultados.add(execucao.executar(1, date));
        //}
        
        //File file = new File(System.getProperty("user.dir"));
        //File parent = new File(file.getParent(),"\\MLP3\\" + dateFormat.format(date) + "\\MLP3_" + dateFormat.format(date) + "_INFO.txt");
        //FileWriter fileWriter = new FileWriter(parent);
        //PrintWriter printWriter = new PrintWriter(fileWriter);

        //System.out.println(dateFormat.format(date));
        //printWriter.println(dateFormat.format(date));

        //printWriter.println("Entradas\t" + ENTRADA);
        //printWriter.println("Camadas Oculta\t" + OCULTA);
        //printWriter.println("Janela\t" + PREVER);
        //printWriter.println("Erro\t" + ERRO);
        System.out.println("Entradas\t" + ENTRADA);
        System.out.println("Camadas Oculta\t" + OCULTA);
        System.out.println("Janela\t" + PREVER);
        System.out.println("Erro\t" + ERRO);

        double soma = 0;
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);

        for (int x = 0; x < 1; x++) {
            System.out.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            //printWriter.println(x + 1 + "\t" + nf.format(resultados.get(x)));
            soma = soma + resultados.get(x);
        }
        System.out.println("Media\t" + nf.format(soma / 10));
        //printWriter.println("Media\t" + nf.format(soma / 10));
    }
}

package mp10;

/*
 * Encog(tm) Java Examples v3.3
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-examples
 *
 * Copyright 2008-2014 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
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

/**
 * This example is meant to demonstrate how to use the Encog TemporalMLDataSet
 * through the full cycle of building a training set, training a model, and then
 * performing predictions. It also demonstrates how to use multiple values in a
 * time-series with normalization and denormalization. The point of this example
 * is more to show how to structure such a prediction problem, than how to
 * accurately predict sun spot data. This example does get marginally good
 * predictive results, however, sun spot data tends to be very noisy.
 *
 * This example was presented as an Encog FAQ, for more FAQ's see:
 *
 * http://www.heatonresearch.com/faq
 */
public class MLP12 {

    /**
     * Set this to whatever you want to use as your home directory. The example
     * is set to use the current directory.
     */
    public static final File MYDIR = new File(".");

    /**
     * This is the amount of data to use to predict.
     */
    public static final int INPUT_WINDOW_SIZE = 4;

    public static final int CAMADA_OCULTA = 22;

    /**
     * This is the amount of data to actually predict.
     */
    public static final int PREDICT_WINDOW_SIZE = 1;

    public static final double TRAIN_TO_ERROR = 0.01;

    /**
     * Used to normalize the SSN (sun spot number) from a range of 0-300 to 0-1.
     */
    public static NormalizedField normMP = new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0);
    public static NormalizedField normTEMP = new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0);
    public static NormalizedField normUR = new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0);

    public static TemporalMLDataSet initDataSet() {
        // create a temporal data set
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);

        // we are dealing with two columns.
        // The first is the sunspot number. This is both an input (used to
        // predict) and an output (we want to predict it), so true,true.
        TemporalDataDescription MP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true);

        // The second is the standard deviation for the month. This is an
        // input (used to predict) only, so true,false.
        TemporalDataDescription TEMP = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription UR = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        dataSet.addDescription(MP);
        dataSet.addDescription(TEMP);
        dataSet.addDescription(UR);
        return dataSet;
    }

    /**
     * Create and train a model. Use Encog factory codes to specify the model
     * type that you want.
     *
     * @param trainingData The training data to use.
     * @param methodName The name of the machine learning method (or model).
     * @param methodArchitecture The type of architecture to use with that
     * model.
     * @param trainerName The type of training.
     * @param trainerArgs Training arguments.
     * @return The trained model.
     */
    public static MLRegression trainModel(
            MLDataSet trainingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs) {

        // first, create the machine learning method (the model)
        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(methodName, methodArchitecture, trainingData.getInputSize(), trainingData.getIdealSize());

        // second, create the trainer
        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method, trainingData, trainerName, trainerArgs);
        // reset if improve is less than 1% over 5 cycles
        if (method instanceof MLResettable && !(train instanceof ManhattanPropagation)) {
            train.addStrategy(new RequiredImprovementStrategy(500));
        }

        // third train the model
        EncogUtility.trainToError(train, TRAIN_TO_ERROR);

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

            // we need a sequence number to sort the data. Here we just use
            // year * 100 + month, which produces output like "201301" for
            // January, 2013.
            int sequenceNumber = x;

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normMP.normalize(MP));
            point.setData(1, normTEMP.normalize(TEMP));
            point.setData(2, normUR.normalize(UR));
            trainingData.getPoints().add(point);
            x++;
        }
        csv.close();

        // generate the time-boxed data
        trainingData.generate();
        return trainingData;
    }

    public static TemporalMLDataSet predict(File rawFile, MLRegression model) throws IOException {
        // You can also use the TemporalMLDataSet for prediction.  We will not use "generate"
        // as we do not want to generate an entire training set.  Rather we pass it each sun spot 
        // ssn and dev and it will produce the input to the model, once there is enough data.
        TemporalMLDataSet trainingData = initDataSet();
        ReadCSV csv = new ReadCSV(rawFile.toString(), true, ';');
        int x = 0;
        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss-10");
        Date date = new Date();
        FileWriter arq = new FileWriter(dateFormat.format(date) + ".csv");
        PrintWriter gravarArq = new PrintWriter(arq);
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.GERMAN);
        double soma = 0;
        int n = 0;
        while (csv.next()) {
            double MP = csv.getDouble(0);
            double TEMP = csv.getDouble(1);
            double UR = csv.getDouble(2);

            // do we have enough data for a prediction yet?
            if (trainingData.getPoints().size() >= trainingData.getInputWindowSize()) {
                // Make sure to use index 1, because the temporal data set is always one ahead
                // of the time slice its encoding.  So for RAW data we are really encoding 0.
                MLData modelInput = trainingData.generateInputNeuralData(1);
                MLData modelOutput = model.compute(modelInput);
                double mp = normMP.deNormalize(modelOutput.getData(0));

                gravarArq.print(x + ";" + nf.format(mp) + ";" + nf.format(MP) + ";" + nf.format(Math.abs(((mp - MP)/MP)*100)) + "\n");
                soma = soma + Math.abs(((mp - MP)/MP)*100);
                n++;
                // Remove the earliest training element.  Unlike when we produced training data,
                // we do not want to build up a large data set.  We just add enough data points to produce
                // input to the model.
                trainingData.getPoints().remove(0);
            }

            // we need a sequence number to sort the data. Here we just use
            // year * 100 + month, which produces output like "201301" for
            // January, 2013.
            int sequenceNumber = x;

            TemporalPoint point = new TemporalPoint(trainingData.getDescriptions().size());
            point.setSequence(sequenceNumber);
            point.setData(0, normMP.normalize(MP));
            point.setData(1, normTEMP.normalize(TEMP));
            point.setData(2, normUR.normalize(UR));
            trainingData.getPoints().add(point);
            x++;
        }
        arq.close();
        csv.close();

        System.out.println("Média Erro do " + dateFormat.format(date) + " = " + nf.format((soma / n)));

        // generate the time-boxed data
        trainingData.generate();
        return trainingData;
    }

    /**
     * The main method.
     *
     * @param args The arguments.
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        // Step 1. Download sun spot data from NASA.
        File arquivoTeste = new File(MYDIR, "10_teste.csv");
        File arquivoTreinamento = new File(MYDIR, "10_treinamento.csv");

        // Step 2. Create training data
        TemporalMLDataSet trainingData = createTraining(arquivoTreinamento);

            // Step 3. Create and train the model.
        // All sorts of models can be used here, see the XORFactory
        // example for more info.
        MLRegression model = trainModel(
                trainingData,
                MLMethodFactory.TYPE_FEEDFORWARD,
                "?:B->SIGMOID->" + CAMADA_OCULTA + ":B->SIGMOID->?",
                MLTrainFactory.TYPE_RPROP,
                "");

        // Now predict
        predict(arquivoTeste, model);

        Encog.getInstance().shutdown();
    }
}
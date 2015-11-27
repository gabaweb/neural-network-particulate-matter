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
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

public class Run {

    public static void main(String[] args) throws IOException {
        Date date = new Date();
        DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        ArrayList<NormalizedField> normalizations = new ArrayList<>();
        ArrayList<Double> results = new ArrayList<>();
        FileWriter arq = new FileWriter(new File(new File(".."), "\\results\\" + dateFormat.format(date) + "\\" + dateFormat.format(date) + "_INFO.txt"));
        PrintWriter gravarArq = new PrintWriter(arq);

        boolean input = true;
        int variables = 1;
        int repetitions = 20;
        int inputWindowSize = 4;
        int hiddenLayerNeurons = 22;
        int predictWindowSize = 1;
        String variable = "MP25";

        normalizations.add(new NormalizedField(NormalizationAction.Normalize, "MP", 100, 0, 1, 0));
        //normalizations.add(new NormalizedField(NormalizationAction.Normalize, "TEMP", 50, 0, 1, 0));
        //normalizations.add(new NormalizedField(NormalizationAction.Normalize, "UR", 100, 0, 1, 0));
        //normalizations.add(new NormalizedField(NormalizationAction.Normalize, "VV", 10, 0, 1, 0));

        for (int x = 1; x <= repetitions; x++) {
            results.add(new MLP(input,
                    variables,
                    inputWindowSize,
                    hiddenLayerNeurons,
                    predictWindowSize,
                    normalizations,
                    variable,
                    x,
                    date).execute());
        }

        System.out.println(dateFormat.format(date));
        gravarArq.println(dateFormat.format(date));
        System.out.println("inputWindowSize\t" + inputWindowSize);
        gravarArq.println("inputWindowSize\t" + inputWindowSize);
        System.out.println("hiddenLayerNeurons\t" + hiddenLayerNeurons);
        gravarArq.println("hiddenLayerNeurons\t" + hiddenLayerNeurons);
        System.out.println("predictWindowSize\t" + predictWindowSize);
        gravarArq.println("predictWindowSize\t" + predictWindowSize);

        double soma = 0;

        for (int x = 0; x < repetitions; x++) {
            System.out.println(x + 1 + "\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(results.get(x)));
            gravarArq.println(x + 1 + "\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(results.get(x)));
            soma = soma + results.get(x);
        }

        System.out.println("Media\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(soma / repetitions));
        gravarArq.println("Media\t" + NumberFormat.getNumberInstance(Locale.GERMAN).format(soma / repetitions));

        arq.close();
    }

}

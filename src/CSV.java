
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;

public class CSV {

    public ArrayList<ArrayList<String>> dados;
    public ArrayList<ArrayList<Double>> dadosTratados;

    public CSV(String arquivo) throws FileNotFoundException, IOException {

        this.dados = new ArrayList<>();
        this.dadosTratados = new ArrayList<>();

        //LER
        BufferedReader br;
        br = new BufferedReader(new FileReader(arquivo));

        for (String linha = br.readLine(); linha != null; linha = br.readLine()) {

            String[] colunas = linha.split(";");
            ArrayList<String> colunasList = new ArrayList<>(Arrays.asList(colunas));
            dados.add(colunasList);
        }

    }

    public ArrayList<Double> mediaColuna(int coluna) throws NumberFormatException {
        ArrayList<Double> resultado;
        resultado = new ArrayList<>();
        int x = -1;

        double soma;
        int n;

        while (x < dados.size() - 1) {

            soma = 0;
            n = 0;

            do {
                x++;

                if (!dados.get(x).get(coluna).equals("#")) {
                    soma = soma + Double.valueOf(dados.get(x).get(coluna));
                    n++;
                }

            } while ((x < dados.size()) && (!dados.get(x).get(1).equals("24:00")));

            if (n == 0) {
                resultado.add(-999.99);

            } else {
                resultado.add(soma / n);

            }
        }

        return resultado;
    }

    public ArrayList<Double> tratarColuna(ArrayList<Double> coluna) {
        //TRATAR DIAS FALTANTES
        double ultimo = -999.99;
        double proximo;
        int y;
        int x;

        for (x = 0; x < coluna.size(); x++) {

            if (coluna.get(x) == -999.99) {

                y = x + 1;

                while (coluna.get(y) == -999.99) {
                    y++;
                }

                proximo = coluna.get(y);

                if (ultimo == -999.99) {

                    coluna.set(x, proximo);

                } else {

                    coluna.set(x, (ultimo + proximo) / 2);

                }

            } else {
                ultimo = coluna.get(x);
            }

        }
        return coluna;
    }

    public void adicionarColuna(ArrayList<Double> coluna) throws IOException {
        
        if (dadosTratados.isEmpty()) {
            ArrayList<Double> linha;
            for (Double valor : coluna) {
                linha = new ArrayList<>();
                linha.add(valor);
                dadosTratados.add(linha);
            }
        } else {
            for (int x = 0; x < coluna.size(); x++) {
                dadosTratados.get(x).add(coluna.get(x));
            }
        }

    }
    

    public void salvar() throws IOException{
    //SALVAR
        FileWriter arq = new FileWriter("tratado.csv");
        PrintWriter gravarArq = new PrintWriter(arq);
        int x;
        int y;

        for (x = 0; x < dadosTratados.size(); x++) {
            for (y = 0; y < dadosTratados.get(x).size(); y++) {
                gravarArq.print(dadosTratados.get(x).get(y) + ";");
            }
            gravarArq.print("\n");
        }
        arq.close();
    }


    public static void main(String args[]) throws FileNotFoundException, IOException, ParseException {

        //TESTAR
        CSV csv = new CSV("dados.csv");
        csv.adicionarColuna(csv.tratarColuna(csv.mediaColuna(2)));
        csv.adicionarColuna(csv.tratarColuna(csv.mediaColuna(3)));
        csv.adicionarColuna(csv.tratarColuna(csv.mediaColuna(4)));
        //for (int x = 0; x < csv.dados.size(); x++){
        //    System.out.println(x+" "+csv.dados.get(x).get(5));
        //}
        csv.adicionarColuna(csv.tratarColuna(csv.mediaColuna(5)));
        csv.salvar();

    }

}

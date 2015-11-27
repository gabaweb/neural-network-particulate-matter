import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.htmlunit.HtmlUnitDriver;
import org.openqa.selenium.support.ui.Select;

public class Qualar {
    
    public static void main(String[] args) {
        // Create a new instance of the html unit driver
        // Notice that the remainder of the code relies on the interface, 
        // not the implementation.
        WebDriver driver = new HtmlUnitDriver();

        // And now use this to visit Google
        driver.get("http://qualar.cetesb.sp.gov.br/qualar/home.do");
        
        WebElement form1 = driver.findElement(By.id("segurancaForm"));
        
        form1.findElement(By.name("cetesb_login")).sendKeys("gabrielbarros");
        form1.findElement(By.name("cetesb_password")).sendKeys("qetwry");
        form1.findElement(By.name("enviar")).submit();
        
        System.out.println(driver.getPageSource());
        
        driver.get("http://qualar.cetesb.sp.gov.br/qualar/exportaDadosAvanc.do?method=pesquisarInit");
        
        WebElement form2 = driver.findElement(By.name("exportaDadosAvancForm"));
        
        form2.findElement(By.name("dataInicialStr")).sendKeys("24/09/2015");
        form2.findElement(By.name("dataFinalStr")).sendKeys("24/10/2015");
        new Select(form2.findElement(By.name("estacaoVO.nestcaMonto"))).selectByVisibleText("Campinas-V.União");
        for (WebElement x : form2.findElements(By.name("nparmtsSelecionados"))){
        
            if (x.getAttribute("value") == "57")
            x.click();
        }
        
        form2.findElement(By.name("btnPesquisar")).click();
       
        System.out.println(driver.getPageSource());

        driver.quit();
    }
    
}

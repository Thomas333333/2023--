package com.example.demo.utils;

import com.example.demo.pojo.Content;
import com.example.demo.pojo.QApair;
import com.hankcs.hanlp.seg.common.Term;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.TermQueryBuilder;
import org.elasticsearch.index.reindex.DeleteByQueryRequest;
import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;

// Boilerpipe库，利用决策树算法解析页面的主题内容
//import de.l3s.boilerpipe.sax.HTMLDocument;
//import de.l3s.boilerpipe.sax.HTMLFetcher;
//import org.xml.sax.InputSource;

//正则化匹配包，用于匹配答案所在句子
import java.util.concurrent.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.util.ArrayList;
import java.util.List;

import com.hankcs.hanlp.HanLP;

@Component

public class HtmlParseUtil {
    public List<QApair> parseResults(String keywords) throws IOException, InterruptedException {
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("bing", true);
        DeleteByQueryRequest deleteByQueryRequest = new DeleteByQueryRequest("question_answer");
        deleteByQueryRequest.setQuery(termQueryBuilder);
        RestHighLevelClient client = new RestHighLevelClient(RestClient.builder(new HttpHost("127.0.0.1", 9200, "http")));
        client.deleteByQuery(deleteByQueryRequest, RequestOptions.DEFAULT);

        ArrayList<QApair> resultList = new ArrayList<>();
        ExecutorService executorService = Executors.newFixedThreadPool(10);
        List<Callable<QApair>> tasks = new ArrayList<>();
        int startId = 2000;
        for (int num = 0; num < 20; num = num + 10) {
            String str_num = String.valueOf(num);
            String url = "https://cn.bing.com/search?q=" + URLEncoder.encode(keywords.toLowerCase(), "utf-8") + "&first=" + str_num;
            //解析网页 (Jsoup返回Document就是浏览器Document对象)
            Connection connect = Jsoup.connect(url);
            connect.header("authority", "cn.bing.com");
            connect.header("method", "POST");
            connect.header("path", "/fd/ls/lsp.aspx");
            connect.header("scheme", "https");
            connect.header("accept", "*/*");
            connect.header("accept-language", "zh-CN,zh;q=0.9");
            connect.header("cookie", "MUID=23797506BDD56A35288164E6B9D56E71; MUIDB=23797506BDD56A35288164E6B9D56E71; SRCHD=AF=CONBDF; SRCHUID=V=2&GUID=E1FF121D7FFB463CB0904590A4894A13&dmnchg=1; _UR=QS=0&TQS=0; MicrosoftApplicationsTelemetryDeviceId=98407e2f-9989-4b4e-830f-038b15966d5c; _clck=m26312|1|f99|0; ANON=A=1024EDDBA94FDC448F158DA8FFFFFFFF&E=1c01&W=1; _U=19XjmT7_j7D4b2rP3RTyeIIqLTejdGhfnnl75yjLdmeAGNffm1Eh6xU3Ck9MyKx53sg6o4ZrzHUJZTudhs2eMOBp5LcCfgcQVB0-pi5RR1bb5XQNUbJBuCV2M0Rzb89Fly2-7gfQifebeK5vrderWc4BWxNOB-btRXifiYtDVQXP_Avnai8UDes_J1CnzQxwKyJEi6-_zdGkJUkCZ6EPNAUddFE_58DUvvPzb8oaCs6U; WLID=t86IbsWLqvjyKWHgpmxKnMhT0T6YMEkbXEKzGSkjULtOYDXnxoe/G1eY2gdZfUsxBqe9XEzl6wR/eRT2P5MyeYDMN5SRmCb2U29d8yAn3wc=; _EDGE_S=SID=29126742003C6A330E727446017F6B93; SUID=A; SRCHS=PC=COS2; WLS=C=8ef47f165018a8ef&N=晨旭; _HPVN=CS=eyJQbiI6eyJDbiI6MzcsIlN0IjoyLCJRcyI6MCwiUHJvZCI6IlAifSwiU2MiOnsiQ24iOjM3LCJTdCI6MCwiUXMiOjAsIlByb2QiOiJIIn0sIlF6Ijp7IkNuIjozNywiU3QiOjEsIlFzIjowLCJQcm9kIjoiVCJ9LCJBcCI6dHJ1ZSwiTXV0ZSI6dHJ1ZSwiTGFkIjoiMjAyMy0wNS0wMVQwMDowMDowMFoiLCJJb3RkIjowLCJHd2IiOjAsIkRmdCI6bnVsbCwiTXZzIjowLCJGbHQiOjAsIkltcCI6MTE5fQ==; USRLOC=HS=1&ELOC=LAT=30.542322158813477|LON=114.37010192871094|N=武昌区，湖北省|ELT=4|; _SS=PC=COS2&SID=29126742003C6A330E727446017F6B93&R=1213&RB=1213&GB=0&RG=0&RP=1213; SRCHUSR=DOB=20220730&T=1682933736000&TPC=1682929631000&POEX=W; ipv6=hit=1682937425228&t=4; SNRHOP=I=&TS=; _FS=CTL=CT3334466&CTT=D072322-N0640AF30FAE38D3; ABDEF=V=13&ABDV=13&MRNB=1682935977886&MRB=0; _RwBf=ilt=42&ihpd=2&ispd=0&rc=1213&rb=1213&gb=0&rg=0&pc=1213&mtu=0&rbb=0.0&g=0&cid=&clo=0&v=8&l=2023-05-01T07:00:00.0000000Z&lft=0001-01-01T00:00:00.0000000&aof=0&o=0&p=bingcopilotwaitlist&c=MY00IA&t=3796&s=2023-02-19T04:34:14.2860630+00:00&ts=2023-05-01T10:12:57.9400730+00:00&rwred=0&wls=2&lka=0&lkt=0&TH=&mta=0&e=wcudA7o3A9185W3doUtfcHUnXe3lV5y_nPR48NUPMp62x5qUP4QVrUB6SZDX2bcuoewDvE3U6RP0xtEoNQkd_2sLmSrvx4j4Zvtp_T2EKf8&A=; SRCHHPGUSR=SRCHLANG=zh-Hans&BRW=NOTP&BRH=M&CW=827&CH=802&SW=1536&SH=864&DPR=1.1&UTC=480&DM=1&PV=10.0.0&WTS=63818075848&HV=1682935978&BZA=0&SCW=1164&SCH=4571&PRVCW=1707&PRVCH=802");
            connect.header("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36");
            connect.header("sec-ch-ua-full-version-list", "\"Chromium\";v=\"112.0.5615.138\", \"Google Chrome\";v=\"112.0.5615.138\", \"Not:A-Brand\";v=\"99.0.0.0\"");
            Document document = connect.get();
            //所有在js中能使用的方法,这里都能用
            Element element = document.getElementById("b_content");
            //获取所有li元素
            Elements elements = element.getElementsByClass("b_algo");
            int id=startId+num;
            //多线程加速
            for (Element el: elements){
                id+=1;
                resultList.add(parseOneElement(el, keywords, id));
            }
//            for (Element el : elements){
//                id +=1;
//                int finalId = id;
//                tasks.add(()->parseOneElement(el, keywords, finalId));
//            }
//            List<Future<QApair>> futures = executorService.invokeAll(tasks);
//            for (Future<QApair> future : futures) {
//                try {
//                    resultList.add(future.get());
//                } catch (InterruptedException | ExecutionException e) {
//                    e.printStackTrace();
//                }
//            }

        }
        executorService.shutdown();
        return resultList;
    }

    public QApair parseOneElement(Element el, String keywords, int id) {

        String question = el.getElementsByTag("h2").eq(0).text();
        String answer = el.getElementsByClass("sh_favicon").eq(0).attr("href");
        String datePattern = "网页\\d+年\\d+月\\d+日";
        String context = el.getElementsByTag("p").eq(0).text().toLowerCase().replaceAll(datePattern, "").replaceAll("网页", "").replaceAll(" · ", "");
        Elements caption_el = el.getElementsByTag("p").eq(0);

        //创建all_answers
        Elements emphasis_el = caption_el.select("Strong");
        List<String> emphasis_text = new ArrayList<>();
        //如果没有strong元素的文本，就自己分词创建
        if (emphasis_el.size() == 0) {
            List<Term> tokenizedKeywords = HanLP.segment(keywords.toLowerCase());
            //获得所有分词后的字符串
            List<String> tokenizendKeywordsString = new ArrayList<>();
            for (Term term : tokenizedKeywords) {
                tokenizendKeywordsString.add(term.word);
            }
            emphasis_text.addAll(tokenizendKeywordsString);
        } else {
            for (int i = 0; i < emphasis_el.size(); i++) {
                Element e = emphasis_el.get(i);
                emphasis_text.add(e.text());
            }
        }
        String[] strongList = new String[emphasis_text.size()];
        for (int i = 0; i < emphasis_text.size(); i++) {
            strongList[i] = emphasis_text.get(i);
        }
        //去重
        Set<String> set = new LinkedHashSet<>(Arrays.asList(strongList));
        List<String> emphasis_text_list = new ArrayList<>(set);
        List<String> all_answers = new ArrayList<String>();

        for (int j = 0; j < emphasis_text_list.size(); j++) {
            //正则式匹配 获取所有答案候选项
            String strong = emphasis_text_list.get(j);
            Pattern pattern = Pattern.compile("[^。]*" + strong + "[^。]*。+");
            Matcher matcher = pattern.matcher(context);
            while (matcher.find()) {
                all_answers.add(matcher.group());
            }
        }

        //再次去重
        Set<String> answer_set = new HashSet<>(all_answers);
        List<String> unique_answers = new ArrayList<>(answer_set);

        //解析搜索结果的链接
        System.out.println(answer);
//        URL res_url = new URL(href);
//
//        HTMLDocument h_doc = HTMLFetcher.fetch(res_url);
//        InputSource inputSource = new InputSource(h_doc.toInputSource().getByteStream());
//        inputSource.setEncoding("UTF-8");
        //解析所有网页，但是比较慢
//                TextDocument t_doc = new BoilerpipeSAXInput(inputSource).getTextDocument();
//                String doc_title = t_doc.getTitle();
//                String context = ArticleExtractor.INSTANCE.getText(t_doc);

        //整合内容，方便写入es
        QApair qApair = new QApair();
        qApair.setQuestion(question);
        qApair.setAll_answers(emphasis_text_list.toArray(new String[0]));
        qApair.setAnswer(answer);
        qApair.setContext(context);
        qApair.setAll_answers(unique_answers.toArray(new String[0]));
        qApair.setBing(true);
        qApair.setId(id);
        return qApair;
    }
}
